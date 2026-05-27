// feedbackTable.js
function extractPoseMetricsFromLabels(labels) {
  const metrics = {};

  for (const label of labels) {
    const pts = label.points || [];
    const angle = label.angle_deg;
    const distance = label.distance_cm;

    // Handle horizontal reach explicitly
    if (label.type === "horizontal_reach" && distance != null) {
      const isRight = (label.label || "").toLowerCase().includes("right");
      const suffix = isRight ? "_right" : "_left";
      metrics[`horizontal_reach_cm${suffix}`] = Math.round(distance * 100) / 100;
      continue;
    }

    const name = getMetricNameFromPoints(pts);
    if (!name) continue;

    const isRight = (label.label || "").toLowerCase().includes("right");
    const suffix = isRight ? "_right" : "_left";

    if (angle != null) {
      metrics[`${name}_angle${suffix}`] = angle;
    } else if (distance != null) {
      metrics[`${name}_distance${suffix}`] = distance;
    }
  }

  return metrics;
}

function extractFeedbackMetrics(metricEntries) {
  const result = {};

  metricEntries.forEach(entry => {
    if ("angle_deg" in entry && Array.isArray(entry.points) && entry.points.length === 3) {
      const [a, b, c] = entry.points;
      const key = `angle_${a}_${b}_${c}`;
      result[key] = parseFloat(entry.angle_deg.toFixed(1));
    } else if ("distance" in entry && typeof entry.distance.value === "number") {
      // Optional: Use entry.name or build a key from points if name is missing
      const key = entry.name || `distance_${entry.points?.join("_") || "unknown"}`;
      result[key] = parseFloat(entry.distance.value.toFixed(2));
    }
  });

  return result;
}


function normalizeFeedbackValue(value) {
  return (value || "").trim().toLowerCase();
}

function getSelectedFeedbackBikeType() {
  const analysisBikeType = document.getElementById("bikeType");
  const feedbackBikeType = document.getElementById("filterBike");
  return normalizeFeedbackValue(analysisBikeType?.value || feedbackBikeType?.value || "road");
}

function getSelectedFeedbackRidingStyle() {
  const analysisRidingStyle = document.getElementById("ridingStyle");
  const feedbackRidingStyle = document.getElementById("filterStyle");
  return normalizeFeedbackValue(analysisRidingStyle?.value || feedbackRidingStyle?.value || "casual");
}

function setSelectValueIfOptionExists(select, value) {
  if (!select || !value) return;
  const normalizedValue = normalizeFeedbackValue(value);
  const option = Array.from(select.options).find(opt => normalizeFeedbackValue(opt.value || opt.textContent) === normalizedValue);
  if (option) select.value = option.value;
}

function syncFeedbackControlsFromAnalysis() {
  const analysisBikeType = document.getElementById("bikeType");
  const analysisRidingStyle = document.getElementById("ridingStyle");
  setSelectValueIfOptionExists(document.getElementById("filterBike"), analysisBikeType?.value);
  setSelectValueIfOptionExists(document.getElementById("filterStyle"), analysisRidingStyle?.value);
}

function loadFeedback() {
  syncFeedbackControlsFromAnalysis();

  const poseMetrics = window.latestPoseMetricsSubset || window.latestPoseMetrics || {};
  const preferredUnit = window.measurementUnit || "in";

  let metricsForFeedback;

  // ✅ Use all metrics when all_fit_points is selected
  if (poseMetrics.all_fit_points && Object.keys(poseMetrics).length === 1) {
    metricsForFeedback = window.latestPoseMetrics;  // Use ALL accumulated layers
  } else if (Array.isArray(poseMetrics)) {
    metricsForFeedback = extractFeedbackMetrics(poseMetrics);
  } else if (Array.isArray(poseMetrics.distances)) {
    metricsForFeedback = extractFeedbackMetrics(poseMetrics.distances);
  } else {
    metricsForFeedback = poseMetrics;
  }

  console.debug("[DEBUG] loadFeedback() Metrics for feedback:", metricsForFeedback);

  const bikeType = getSelectedFeedbackBikeType();
  const ridingStyle = getSelectedFeedbackRidingStyle();
  console.log("[FEEDBACK DEBUG] bike_type:", bikeType, "style:", ridingStyle);

  const payload = {
    metrics: metricsForFeedback,
    unit: preferredUnit,
    bike_type: bikeType,
    style: ridingStyle
  };

  fetch("/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  })
    .then(res => res.json())
    .then(data => {
      console.log("[DEBUG] Received feedback data:", data);
      renderFeedbackTable(data.feedback);
      window.latestFeedback = data.feedback;    // added to have handle to latest feedback for 'export' btn
    })
    .catch(err => {
      console.error("Error loading feedback:", err.message);
    });
}

/*
  // ✅ Inject distance metrics from all_fit_points into flat structure
  const allFit = poseMetrics["all_fit_points"];
  if (allFit && Array.isArray(allFit.distances)) {
    for (const d of allFit.distances) {
      const pts = d.points || [];
      if (pts.length !== 2) continue;

      const [a, b] = pts;
      let side = "";
      if ([11, 13, 15, 5].includes(a) || [11, 13, 15, 5].includes(b)) {
        side = "left";
      } else if ([12, 14, 16, 6].includes(a) || [12, 14, 16, 6].includes(b)) {
        side = "right";
      }

      const key = `all_fit_points_${side}_${a}_${b}`;
      metricsForFeedback[key] = {
        distances: [d]
      };
    }
  }

  console.debug("[DEBUG] loadFeedback() Metrics for feedback:", metricsForFeedback);

  const payload = {
    metrics: metricsForFeedback,
    unit: preferredUnit,
    bike_type: bikeType,
    style: ridingStyle
  };

  console.debug("[DEBUG] loadFeedback() Feedback payload:", payload);

  fetch("/feedback", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  })
    .then(res => res.json())
    .then(data => {
      console.log("[DEBUG] Received feedback data:", data);
      renderFeedbackTable(data.feedback);
    })
    .catch(err => {
      console.error("Error loading feedback:", err);
    });
}
*/

function getMetricNameFromPoints(pts) {
  const key = pts.slice().sort((a, b) => a - b).join("-");

  const pointMap = {
    "5-11-13": "hip",
    "11-13-15": "knee",
    "11-5-6": "torso",
    "5-9": "leg_length",
    "11-9": "saddle_to_bar",
    "6-10": "leg_length",
    "12-14-16": "knee",
    "6-12-14": "hip",
    "12-5-6": "torso",
    "12-10": "saddle_to_bar",
    "12-6-8": "shoulder",
    "6-8": "arm_reach"
    // Extend this list as needed
  };

  return pointMap[key] || null;
}

function renderFeedbackTable(rows) {
  const table = $("#feedbackTable").DataTable();
  table.clear(); // Clear existing data

  if (!rows || rows.length === 0) {
    table.row.add(["No feedback available.", "", "", "", ""]);
    table.draw();
    return;
  }

  const displayed = [];
const seen = new Set(); // hold temp row(s) to check for dups

for (let i = 0; i < rows.length; i++) {
  const row = rows[i];

  const label = row.label || row.metric || "–";
  const value = row.value !== undefined ? row.value : "–";
  let target = row.target || row.range || "–";
  const status = row.status || "–";
  const explanation = row.explanation || "–";

  const key = `${label}|${value}|${status}`; // build temp key to check if dup
  console.log("keyyyyy=" + key);

  // skip duplicates
  if (seen.has(key))
  {
    console.log("SKIPPING -keyyyyy=" + key);
    continue;
  }

  // record this key so future rows with the same triple get skipped
  seen.add(key);

    displayed.push(row);          // <-- push here dy 0703 -feedback key


  // Sanitize degree symbol to ensure it renders as UTF-8
  if (typeof target === "string" && target.includes("°")) {
    target = target.replace(/°/g, "\u00B0");  // Safe Unicode representation
  }
  console.log("Final target string:", target);

  // Add the row to your DataTable
  table.row.add([
    $("<span>").text(label).prop("outerHTML"),
    $("<span>").text(value).prop("outerHTML"),
    $("<span>").text(target).prop("outerHTML"),
    $("<span>").text(status).prop("outerHTML"),
    $("<span>").text(explanation).prop("outerHTML")
  ]);
}


  table.draw();

  // dy -0703 - add key to each row in table to allow click
    // **Here**: tag each <tr> with the metric key
  table.rows().every(function(idx) {
    const tr = this.node();
    const datum = displayed[idx];
    if (datum.key) {
      tr.dataset.metricKey = datum.key;
    }
  });
}



$(document).ready(function () {
  $("#feedbackTable").DataTable();

    // dy - 0703 - Hook up the feedback‐table click - not sure if right place
  let highlightedKey = null;
const tbody = document.querySelector('#feedbackTable tbody');

if (tbody) {
  tbody.addEventListener('click', e => {
    const tr = e.target.closest('tr');
    if (!tr) return;

    // clear any old highlights
    document.querySelectorAll('.highlighted-overlay')
            .forEach(el => el.classList.remove('highlighted-overlay'));

    highlightedKey = tr.dataset.metricKey;
    if (!highlightedKey) return;

    // highlight all SVG bits with that metricKey
    document.querySelectorAll(`[data-metric-key="${highlightedKey}"]`)
            .forEach(el => el.classList.add('highlighted-overlay'));
  });

  // 0703-dy click handler to feedback row
  console.log("...feedbackTable.js - adding click handler to row...")
  /*
tbody.addEventListener('click', e => {
  const tr = e.target.closest('tr');
  if (!tr) return;
  document.querySelectorAll('.highlighted-overlay')
          .forEach(el=>el.classList.remove('highlighted-overlay'));
  const key = tr.dataset.metricKey;
  document.querySelectorAll(`[data-metric-key="${key}"]`)
          .forEach(el=>el.classList.add('highlighted-overlay'));
   let txt=`[data-metric-key="${key}"]`
  console.log("....feedbackTable.js - CLICKED key="+txt)
});*/


  tbody.addEventListener('click', e => {
    const tr = e.target.closest('tr');
    if (!tr) return;

    // 1) Remove prior row highlights
    document.querySelectorAll('#feedbackTable tbody tr.selected-row')
      .forEach(r => r.classList.remove('selected-row'));

    // 2) Add to this row
    tr.classList.add('selected-row');

    // 3) Remove any old SVG highlights
    document.querySelectorAll('.highlighted-overlay')
      .forEach(el => el.classList.remove('highlighted-overlay'));

    // 4) Highlight matching SVG bits
    const key = tr.dataset.metricKey;
    document.querySelectorAll(`[data-metric-key="${key}"]`)
      .forEach(el => el.classList.add('highlighted-overlay'));

         let txt=`[data-metric-key="${key}"]`
  console.log("....feedbackTable.js - CLICKED key="+txt)
  });


}






});
