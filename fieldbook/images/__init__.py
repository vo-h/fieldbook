"""Environment images for cloud runs (cloud-runs.md §4–5), end to end: what an
image is built from and its tag (`spec`), the ECR registry (`registry`), and building and
finding it (`build`).

One image per environment, not per run: it holds Python and the packages in `uv.lock`, and
is tagged with a hash of everything it is built from, so the same environment always maps
to the same image and a code change never needs a build. The code comes from the run's
snapshot when the run starts.
"""

from fieldbook.images.build import Image, image
from fieldbook.images.spec import Environment, ImageError, ImageMissing, environment

__all__ = ["Environment", "Image", "ImageError", "ImageMissing", "environment", "image"]
