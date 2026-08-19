#!/usr/bin/perl
# Persistent Image::ExifTool server.
#
# REST.py previously spawned a new "exiftool" process (a Perl interpreter
# plus the Image::ExifTool module) for every single request, via either the
# PyExifTool "-stay_open" wrapper opened-and-closed per call, or a bare
# subprocess.run() for the bytes-upload path. Both pay full process-startup
# cost (measured ~50ms of the ~53-56ms per request) on every call. This
# daemon loads Image::ExifTool exactly once and answers requests over a Unix
# domain socket for the life of the metadata service.
#
# Protocol (length-prefixed, no shell/text framing so it's exact for binary
# image data): request is [1 byte mode][4 bytes big-endian length][payload].
# Response is [4 bytes big-endian length][JSON object body]. Both modes
# produce the same output shape - flat (ungrouped) tag names, human-readable
# print-converted values, matching plain "exiftool -j" - they only differ in
# where the image comes from:
#
#   'P' - payload is a UTF-8 file path (an on-disk file REST.py already has
#         a legitimate path to).
#   'D' - payload is raw image bytes, handed directly to
#         Image::ExifTool::ImageInfo() as an in-memory scalar reference -
#         never written to disk.
#
# These two modes used to produce different output shapes (one grouped/raw,
# one flat/human-readable), inherited from how REST.py happened to invoke
# exiftool before this daemon existed. Unified onto one shape since nothing
# in this codebase consumes either format specifically, and having a
# service's two extraction paths silently diverge in output shape was itself
# the bug.
#
# Single-threaded accept loop by design: request handling is a few
# milliseconds, and serializing concurrent requests through one warm
# process is far simpler (no locking, no fork/zombie-reaping) than the
# alternative, at the cost of not parallelizing metadata extraction across
# concurrent Flask requests.
use strict;
use warnings;
use Image::ExifTool;
use IO::Socket::UNIX;

$| = 1;

my $socket_path = shift @ARGV or die "Usage: $0 <socket_path>\n";
unlink $socket_path if -e $socket_path;

my $server = IO::Socket::UNIX->new(
    Type   => SOCK_STREAM,
    Local  => $socket_path,
    Listen => 128,
) or die "exiftool_daemon: cannot bind $socket_path: $!\n";

my $running = 1;
$SIG{TERM} = $SIG{INT} = sub { $running = 0; };

print STDERR "exiftool_daemon: listening on $socket_path\n";

while ($running) {
    my $client = $server->accept();
    next unless $client;
    binmode $client;
    eval { handle_client($client); };
    warn "exiftool_daemon: request failed: $@" if $@;
    close $client;
}

close $server;
unlink $socket_path;
exit 0;

sub read_exact {
    my ($fh, $len) = @_;
    return '' if $len == 0;
    my $buf = '';
    while (length($buf) < $len) {
        my $chunk;
        my $n = sysread($fh, $chunk, $len - length($buf));
        return undef unless $n;
        $buf .= $chunk;
    }
    return $buf;
}

sub handle_client {
    my ($client) = @_;

    my $mode = read_exact($client, 1);
    return unless defined $mode;
    my $len_buf = read_exact($client, 4);
    return unless defined $len_buf;
    my $len = unpack('N', $len_buf);
    return if $len < 0 || $len > 200 * 1024 * 1024;

    my $payload = read_exact($client, $len);
    return unless defined $payload;

    my %result;
    eval {
        my $et = Image::ExifTool->new;
        # Flat (ungrouped) tag names, human-readable print-converted values -
        # matches plain "exiftool -j" output. Duplicates => 0 makes
        # ImageInfo() resolve same-named tags via ExifTool's own tag
        # priority rules, same as the CLI's default (ungrouped) mode.
        $et->Options(Duplicates => 0);
        my $info = $mode eq 'P' ? $et->ImageInfo($payload) : $et->ImageInfo(\$payload);
        for my $tag (keys %$info) {
            next if $tag eq 'SourceFile';
            $result{$tag} = $info->{$tag};
        }
    };
    if ($@) {
        my $err = $@;
        $err =~ s/\s+$//;
        %result = ('error' => "ExifTool extraction failed: $err");
    }

    my $json = to_json_like_exiftool(\%result);
    print $client pack('N', length($json)) . $json;
}

# Mirrors exiftool's own EscapeJSON()/FormatJSON() number-vs-string rule
# (see the "sub EscapeJSON" in the exiftool CLI script) so Python's
# json.loads() gets identical types to what "exiftool -j -G -n" produces.
sub is_json_number {
    my ($v) = @_;
    return $v =~ /^-?(\d|[1-9]\d{1,14})(\.\d{1,16})?(e[-+]?\d{1,3})?$/i;
}

sub escape_json_string {
    my ($s) = @_;
    $s = '' unless defined $s;
    my %esc = ("\\" => "\\\\", "\"" => "\\\"", "\t" => "\\t", "\n" => "\\n", "\r" => "\\r");
    $s =~ s/([\\"\t\n\r])/$esc{$1}/ge;
    $s =~ s/([\x00-\x1f])/sprintf("\\u%04X", ord($1))/ge;
    utf8::encode($s) if utf8::is_utf8($s);
    return '"' . $s . '"';
}

# Tags holding undecoded binary (ICC profile curves, embedded thumbnails,
# etc.) come back from ImageInfo() as a SCALAR reference to the raw bytes,
# not a plain string. Without -b (which we never pass), exiftool's own CLI
# replaces these with a "(Binary data N bytes, ...)" placeholder rather than
# embedding the bytes (see "sub ConvertBinary" in the exiftool CLI script) -
# match that so this never falls through to Perl's default ref
# stringification ("SCALAR(0x...)", a useless, non-deterministic value).
sub stringify_ref {
    my ($v) = @_;
    if (ref($v) eq 'SCALAR') {
        return '(Binary data ' . length($$v) . ' bytes, use -b option to extract)';
    } elsif (ref($v) eq 'ARRAY') {
        return join(', ', map { ref($_) ? stringify_ref($_) : $_ } @$v);
    } elsif (ref($v) eq 'HASH') {
        return join(', ', map { "$_=" . (ref($v->{$_}) ? stringify_ref($v->{$_}) : $v->{$_}) } sort keys %$v);
    }
    return $v;
}

sub to_json_like_exiftool {
    my ($hash) = @_;
    my @parts;
    for my $key (sort keys %$hash) {
        my $v = $hash->{$key};
        $v = stringify_ref($v) if ref($v);
        my $jv;
        if (!defined $v) {
            $jv = 'null';
        } elsif (!ref($v) && is_json_number($v)) {
            $jv = $v;
        } else {
            $jv = escape_json_string("$v");
        }
        push @parts, escape_json_string($key) . ':' . $jv;
    }
    return '{' . join(',', @parts) . '}';
}
