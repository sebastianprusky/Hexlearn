importScripts("core.js");

chrome.action.onClicked.addListener(async (tab) => {
  if (!tab.id || !globalThis.PlayLensCore.supportedGameUrl(tab.url)) {
    await chrome.action.setBadgeText({ tabId: tab.id, text: "!" });
    await chrome.action.setBadgeBackgroundColor({ tabId: tab.id, color: "#e0a24a" });
    return;
  }

  try {
    const response = await chrome.tabs.sendMessage(tab.id, { type: "PLAYLENS_TOGGLE" });
    await chrome.action.setBadgeText({ tabId: tab.id, text: response?.armed ? "ON" : "" });
    await chrome.action.setBadgeBackgroundColor({ tabId: tab.id, color: "#5dd39e" });
  } catch (error) {
    console.warn("Hexlearn content script is unavailable", error);
    await chrome.action.setBadgeText({ tabId: tab.id, text: "!" });
    await chrome.action.setBadgeBackgroundColor({ tabId: tab.id, color: "#e05555" });
  }
});
