// assume:
//   - <img id="analyzeImage"> holds the processed base image
//   - <canvas id="overlayCanvas"> draws your annotation layers
//   - window.latestFeedback is an array of your feedback objects
//       e.g. [{ metric, value, target, status, explanation }, …]

// click handler defined in <button...
document.addEventListener('DOMContentLoaded', () => {
    document
      .getElementById('exportBtn')
      .addEventListener('click', exportResults);
});

function exportResults() {
  const baseImg    = document.getElementById('analyzeImage');
  const reachCanvas= document.getElementById('overlayCanvas');
  const svgOverlay = document.getElementById('svgOverlay');

  const w = baseImg.naturalWidth,
        h = baseImg.naturalHeight;
  if (!w || !h) return console.error("Image not loaded");

  // make our off-screen canvas
  const exportCanvas = document.createElement('canvas');
  exportCanvas.width  = w;
  exportCanvas.height = h;
  const ctx = exportCanvas.getContext('2d');

  // draw base + canvas layer
  ctx.drawImage(baseImg, 0, 0, w, h);
  if (reachCanvas?.width) ctx.drawImage(reachCanvas, 0, 0, w, h);

  // helper to do both downloads
  function doDownloads() {
    exportCanvas.toBlob(blob => {
      // — 1) download the PNG
      const imgLink = document.createElement('a');
      imgLink.href     = URL.createObjectURL(blob);
      imgLink.download = 'annotated-composite.png';
      imgLink.click();
      URL.revokeObjectURL(imgLink.href);

      // — 2) download the JSON (if we have it)
      if (Array.isArray(window.latestFeedback) && window.latestFeedback.length) {
      console.log("exportResults: writing json results to file")
        const jsonStr = JSON.stringify(window.latestFeedback, null, 2);
        const fbBlob  = new Blob([jsonStr], { type: 'application/json' });
        const fbLink  = document.createElement('a');
        fbLink.href     = URL.createObjectURL(fbBlob);
        fbLink.download = 'feedback.json';
        fbLink.click();
        URL.revokeObjectURL(fbLink.href);
      } else {
        console.warn("No feedback to save");
      }
    }, 'image/png');
  }

  // if there’s an SVG overlay, rasterize it first
  if (svgOverlay) {
    const clone = svgOverlay.cloneNode(true);
    clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    clone.setAttribute('width',  w);
    clone.setAttribute('height', h);
    const svgData = new XMLSerializer().serializeToString(clone);
    const url     = URL.createObjectURL(new Blob([svgData], {type:'image/svg+xml'}));
    const imgSVG  = new Image();
    imgSVG.onload = () => {
      ctx.drawImage(imgSVG, 0, 0, w, h);
      URL.revokeObjectURL(url);
      doDownloads();
    };
    imgSVG.src = url;
  } else {
    // no SVG → just go straight to downloads
    doDownloads();
  }
}

/* if want CSV instead of json....
         …then in exportResults(), replace the JSON‐blob block with:
        const csv = feedbackToCSV(window.latestFeedback);
        const blob = new Blob([csv], { type: 'text/csv' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'feedback.csv';
        link.click();
        URL.revokeObjectURL(link.href);
*/
function feedbackToCSV(arr) {
  const header = ['Metric','Value','Target','Status','Explanation'];
  const lines = arr.map(o =>
    [o.metric,o.value,o.target,o.status,o.explanation]
      .map(v => `"${String(v).replace(/"/g,'""')}"`)
      .join(',')
  );
  return [ header.join(','), ...lines ].join('\r\n');
}
/*
Make sure you store your feedback in a global (e.g. window.latestFeedback) when you get it back from /feedback.

If you ever want to bundle both files into a ZIP, you can include JSZip and pack both blobs before offering a single download.
*/