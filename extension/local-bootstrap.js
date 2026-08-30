(function bootstrapLocalPlayLens() {
  "use strict";

  const core = globalThis.PlayLensCore;
  if (!core) return;
  const root = document.documentElement;
  const reservation = "local-fallback-pending";
  const reserved = core.claimControllerOwnership(root, reservation);

  const launch = () => window.setTimeout(() => {
    if (!reserved || root.getAttribute(core.CONTROLLER_LOCK_ATTRIBUTE) !== reservation) return;
    root.removeAttribute(core.CONTROLLER_LOCK_ATTRIBUTE);
    const script = document.createElement("script");
    script.src = "/playlens/content.js?v=064";
    script.dataset.playlensLocalController = "true";
    document.body.appendChild(script);
  }, 100);

  if (document.readyState === "complete") launch();
  else window.addEventListener("load", launch, { once: true });
})();
