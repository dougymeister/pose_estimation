// overlayButtonManager.js

// Track currently active overlay buttons
let currentActiveButtons = new Set();

// Toggle button active state visually and in state tracker
function setButtonState(btnId, isActive) {
  const btn = document.getElementById(btnId);
  if (!btn) return;

  if (isActive) {
    btn.classList.add("active-overlay");
    currentActiveButtons.add(btnId);
  } else {
    btn.classList.remove("active-overlay");
    currentActiveButtons.delete(btnId);
  }
}

// Deactivate all overlay buttons and clear tracking
function clearAllButtonStates() {
  currentActiveButtons.forEach(btnId => {
    setButtonState(btnId, false);
  });
  currentActiveButtons.clear();
}

// Get currently active layers based on button IDs
function getActiveLayers() {
  const layers = [];
  currentActiveButtons.forEach(btnId => {
    if (btnId.startsWith("btn-")) {
      const layer = btnId.replace("btn-", "");
      layers.push(layer);
    }
  });
  return layers;
}

// Optional: Helper to activate a group of buttons by ID
function activateButtonSet(btnIdArray) {
  clearAllButtonStates();
  btnIdArray.forEach(btnId => {
    setButtonState(btnId, true);
  });
}
