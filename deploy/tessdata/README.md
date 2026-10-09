Optional: Tesseract language models for building the API image without network access.
Put `ara.traineddata` from [tessdata_best 4.1.0](https://github.com/tesseract-ocr/tessdata_best/tree/4.1.0) here;
otherwise the build downloads it. Either way the build verifies its SHA-256 (see `deploy/Dockerfile`).
Language model files are not tracked by git.
