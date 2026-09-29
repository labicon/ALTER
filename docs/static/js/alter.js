"use strict";

document.querySelectorAll("[data-media]").forEach((slot) => {
  const asset = (window.ALTER_MEDIA || {})[slot.dataset.media];
  if (!asset || !asset.src || !["video", "image"].includes(asset.type)) return;

  const media = document.createElement(asset.type === "video" ? "video" : "img");
  if (asset.type === "video") {
    media.controls = true;
    media.playsInline = true;
    media.preload = slot.closest("details") ? "none" : "metadata";
    media.setAttribute("aria-label", asset.alt);
    if (asset.poster) media.poster = asset.poster;
  } else {
    media.alt = asset.alt;
    media.loading = "lazy";
  }

  // Keep a useful placeholder if a configured file is missing or unsupported.
  const placeholder = slot.firstElementChild;
  media.addEventListener("error", () => {
    slot.replaceChildren(placeholder);
    slot.classList.remove("has-media");
    slot.style.removeProperty("aspect-ratio");
    slot.setAttribute("role", "img");
    slot.setAttribute("aria-label", `${asset.alt} — media unavailable`);
  }, { once: true });
  slot.removeAttribute("role");
  slot.removeAttribute("aria-label");
  if (asset.type === "image" && asset.mobileSrc) {
    const picture = document.createElement("picture");
    const source = document.createElement("source");
    source.media = "(max-width: 680px)";
    source.srcset = asset.mobileSrc;
    picture.append(source, media);
    slot.replaceChildren(picture);
  } else {
    slot.replaceChildren(media);
  }
  slot.classList.add("has-media");
  if (asset.width && asset.height) slot.style.aspectRatio = `${asset.width} / ${asset.height}`;
  media.src = asset.src;
});
