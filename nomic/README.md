# Nomic Embed v1.5 Service

Embeddings-only service using the aligned `nomic-embed-text-v1.5` and
`nomic-embed-vision-v1.5` models. It mirrors the SigLIP service API on port 7808.

```bash
cd nomic && cp .env.sample .env && ./install.sh && ./run.sh
curl localhost:7808/health
curl -X POST localhost:7808/embed/text -H 'Content-Type: application/json' -d '{"terms":["dog","cat"],"task":"search_document"}'
curl -X POST localhost:7808/embed/text -H 'Content-Type: application/json' -d '{"terms":["a dog"],"task":"search_query"}'
curl -X POST localhost:7808/embed/image -F 'file=@image.jpg'
```

Text tasks are `search_document` (default), `search_query`, `classification`, or
`clustering`. Use `search_document` for stored text and `search_query` for a query.
Image and text vectors are L2-normalized in the shared 768-dimensional space.
