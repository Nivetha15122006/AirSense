/**
 * AirSense Interactive Application Controller
 * Comprehensive logic for dual themes, city intelligence, live forecasts,
 * evaluation metrics, and Chart.js visualizations.
 */

let citiesData = [];
let currentCitySummary = null;
let currentPrediction = null;

// Chart Instances
let chartForecastComparison = null;
let chartPollutantLimits = null;
let chartR2 = null;
let chartError = null;
let chartHistorical = null;

// CPCB 24-Hour Safe Limit Guidelines (NAAQS India)
const CPCB_LIMITS = {
  pm25: 60,   // ug/m3
  pm10: 100,  // ug/m3
  no2: 80,    // ug/m3
  so2: 80,    // ug/m3
  co: 2.0,    // mg/m3 (8-hour)
  o3: 100     // ug/m3 (8-hour)
};

const PROMINENT_CHIPS = [
  "Chennai", "Delhi", "Bengaluru", "Mumbai", "Coimbatore",
  "Hyderabad", "Kolkata", "Jaipur", "Lucknow", "Ahmedabad", "Patna", "Chandigarh"
];

// Document Ready Initialization
document.addEventListener("DOMContentLoaded", async () => {
  initTheme();

  // Set default date to today
  const today = new Date().toISOString().split("T")[0];
  const dateInput = document.getElementById("input-date");
  if (dateInput) dateInput.value = today;

  await initApp();
});

/* ==========================================================================
   THEME TOGGLING (LIGHT / DARK MODE)
   ========================================================================== */
function initTheme() {
  const savedTheme = localStorage.getItem("airsense-theme") || "dark";
  applyTheme(savedTheme);
}

function toggleTheme() {
  const isDark = document.documentElement.classList.contains("dark");
  const newTheme = isDark ? "light" : "dark";
  applyTheme(newTheme);
  localStorage.setItem("airsense-theme", newTheme);
}

function applyTheme(theme) {
  const icon = document.getElementById("theme-icon");
  const text = document.getElementById("theme-text");

  if (theme === "dark") {
    document.documentElement.classList.add("dark");
    if (icon) icon.className = "fa-solid fa-sun text-amber-400";
    if (text) text.innerText = "Light Mode";
  } else {
    document.documentElement.classList.remove("dark");
    if (icon) icon.className = "fa-solid fa-moon text-indigo-600";
    if (text) text.innerText = "Dark Mode";
  }

  // Refresh active charts with new theme colors
  setTimeout(() => {
    if (chartForecastComparison) renderForecastComparisonChart();
    if (chartPollutantLimits) renderPollutantLimitsChart();
    if (chartHistorical) loadCityHistoricalTrend();
    if (chartR2 || chartError) loadModelBenchmarks();
  }, 100);
}

function isDarkMode() {
  return document.documentElement.classList.contains("dark");
}

/* ==========================================================================
   TAB NAVIGATION (RESPONSIVE)
   ========================================================================== */
function switchTab(tabId) {
  const tabs = ["forecast", "models", "trends", "error", "methodology"];
  
  tabs.forEach(t => {
    const pane = document.getElementById(`tab-${t}`);
    const dBtn = document.getElementById(`tab-btn-${t}`);
    const mBtn = document.getElementById(`m-tab-btn-${t}`);

    if (pane) {
      if (t === tabId) {
        pane.classList.remove("hidden");
      } else {
        pane.classList.add("hidden");
      }
    }

    if (dBtn) {
      if (t === tabId) dBtn.classList.add("active");
      else dBtn.classList.remove("active");
    }

    if (mBtn) {
      if (t === tabId) mBtn.classList.add("active");
      else mBtn.classList.remove("active");
    }
  });

  // Re-trigger layout for canvas elements
  setTimeout(() => {
    if (tabId === "models") {
      if (chartR2) chartR2.resize();
      if (chartError) chartError.resize();
    } else if (tabId === "trends") {
      if (chartHistorical) chartHistorical.resize();
    } else if (tabId === "forecast") {
      if (chartForecastComparison) chartForecastComparison.resize();
      if (chartPollutantLimits) chartPollutantLimits.resize();
    }
  }, 50);
}

/* ==========================================================================
   APPLICATION INITIALIZATION
   ========================================================================== */
async function initApp() {
  try {
    // 1. Fetch cities list
    const citiesRes = await fetch("/api/cities");
    const citiesJson = await citiesRes.json();
    citiesData = citiesJson.cities || [];

    populateCityControls();

    // Default city: Chennai
    const defaultCity = citiesData.some(c => c.name === "Chennai") ? "Chennai" : (citiesData[0]?.name || "Chennai");
    document.getElementById("input-city").value = defaultCity;

    // Load initial city summary & run initial prediction
    await onCityChanged();

    // Preload background tabs
    loadModelBenchmarks();
    loadErrorAnalysis();

    document.getElementById("api-status-text").innerText = "Online";
  } catch (err) {
    console.error("Initialization failed:", err);
    document.getElementById("api-status-text").innerText = "API Error";
    document.getElementById("api-status-text").parentElement.classList.replace("text-emerald-500", "text-rose-500");
  }
}

function populateCityControls() {
  const citySelect = document.getElementById("input-city");
  const trendSelect = document.getElementById("trend-city-select");
  const chipsContainer = document.getElementById("quick-city-chips");

  if (citiesData && citiesData.length > 0) {
    if (citySelect) {
      citySelect.innerHTML = "";
      citiesData.forEach(city => {
        const opt = document.createElement("option");
        opt.value = city.name;
        opt.textContent = `${city.name} (${city.region.split(" ")[0]})`;
        citySelect.appendChild(opt);
      });
    }

    if (trendSelect) {
      trendSelect.innerHTML = "";
      citiesData.forEach(city => {
        const opt2 = document.createElement("option");
        opt2.value = city.name;
        opt2.textContent = city.name;
        trendSelect.appendChild(opt2);
      });
    }

    if (chipsContainer) {
      chipsContainer.innerHTML = "";
      PROMINENT_CHIPS.forEach(chipName => {
        if (citiesData.some(c => c.name.toLowerCase() === chipName.toLowerCase())) {
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "quick-chip px-3 py-1 rounded-lg text-xs font-semibold bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-cyan-500/20 hover:text-cyan-600 dark:hover:text-cyan-400 border border-slate-200 dark:border-slate-700 transition-all";
          btn.innerHTML = `<i class="fa-solid fa-location-dot mr-1 text-[10px] text-cyan-500"></i>${chipName}`;
          btn.onclick = () => selectQuickCity(chipName);
          chipsContainer.appendChild(btn);
        }
      });
    }
  }
}

function selectQuickCity(cityName) {
  const citySelect = document.getElementById("input-city");
  if (citySelect) {
    citySelect.value = cityName;
    onCityChanged();
  }
}

/* ==========================================================================
   CITY CHANGE & SUMMARY HANDLING
   ========================================================================== */
async function onCityChanged() {
  const cityName = document.getElementById("input-city").value;
  if (!cityName) return;

  // Highlight active chip
  document.querySelectorAll(".quick-chip").forEach(chip => {
    if (chip.textContent.trim().includes(cityName)) {
      chip.classList.add("border-cyan-500", "bg-cyan-500/10", "text-cyan-600", "dark:text-cyan-400");
    } else {
      chip.classList.remove("border-cyan-500", "bg-cyan-500/10", "text-cyan-600", "dark:text-cyan-400");
    }
  });

  // Sync trends tab dropdown
  const trendSelect = document.getElementById("trend-city-select");
  if (trendSelect) trendSelect.value = cityName;

  // Fetch and display city air quality intelligence
  await loadCitySummary(cityName);

  // Auto-fill sliders with city norms
  autoFillCityDefaults();

  // Run prediction
  await handlePredict();

  // Update historical chart if trend tab is active
  loadCityHistoricalTrend();
}

async function loadCitySummary(cityName) {
  try {
    const res = await fetch(`/api/city-summary?city=${encodeURIComponent(cityName)}`);
    if (!res.ok) throw new Error("City summary fetch error");
    const data = await res.json();
    currentCitySummary = data;

    // Update Header Card
    document.getElementById("sum-city-name").innerText = data.city;
    document.getElementById("sum-date-text").innerText = `Latest recorded: ${data.latest_date} • 26 Indian Cities Database`;
    
    // Category Badge
    const catBadge = document.getElementById("sum-category-badge");
    catBadge.innerText = data.latest_category;
    catBadge.style.backgroundColor = data.latest_color;
    catBadge.style.color = ["Good", "Satisfactory", "Moderate"].includes(data.latest_category) ? "#020617" : "#FFFFFF";

    // 4 Key Metrics
    document.getElementById("sum-latest-aqi").innerText = data.latest_aqi;
    document.getElementById("sum-avg-7").innerText = data.avg_7_days;
    document.getElementById("sum-avg-30").innerText = data.avg_30_days;
    document.getElementById("sum-city-mae").innerText = `±${data.evaluation_metrics.city_test_mae}`;

    // Trend Direction & Delta
    const trendIcon = document.getElementById("sum-trend-icon");
    const trendText = document.getElementById("sum-trend-text");
    if (data.trend_direction === "Improved") {
      trendIcon.className = "fa-solid fa-arrow-down text-emerald-500";
      trendText.innerText = `7-Day Trend: Improved by ${Math.abs(data.trend_delta)} AQI`;
    } else if (data.trend_direction === "Worsened") {
      trendIcon.className = "fa-solid fa-arrow-up text-rose-500";
      trendText.innerText = `7-Day Trend: Worsened by ${Math.abs(data.trend_delta)} AQI`;
    } else {
      trendIcon.className = "fa-solid fa-arrow-right text-slate-400";
      trendText.innerText = `7-Day Trend: Stable (Δ ${data.trend_delta} AQI)`;
    }

    // Render Mini Sparkline Bars
    renderSparklineBars(data.sparkline || []);

  } catch (err) {
    console.error("Error loading city summary:", err);
  }
}

function renderSparklineBars(sparkline) {
  const container = document.getElementById("sparkline-bars");
  const label = document.getElementById("sparkline-trend-label");
  if (!container || sparkline.length === 0) return;

  container.innerHTML = "";
  const maxAqi = Math.max(...sparkline.map(s => s.aqi), 150);

  label.innerText = `Range: ${Math.min(...sparkline.map(s => s.aqi))} - ${Math.max(...sparkline.map(s => s.aqi))} AQI`;

  sparkline.forEach(item => {
    const heightPercent = Math.max(15, Math.min(100, (item.aqi / maxAqi) * 100));
    const barColor = getAqiColorHex(item.aqi);

    const col = document.createElement("div");
    col.className = "flex-1 flex flex-col items-center group relative cursor-pointer";
    col.innerHTML = `
      <div class="text-[9px] font-mono text-slate-500 dark:text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity mb-1">${item.aqi}</div>
      <div class="w-full rounded-t-md transition-all group-hover:brightness-110" style="height: ${heightPercent}%; background-color: ${barColor}"></div>
      <div class="text-[9px] font-semibold text-slate-500 dark:text-slate-400 mt-1 whitespace-nowrap">${item.date}</div>
    `;
    container.appendChild(col);
  });
}

function getAqiColorHex(aqi) {
  if (aqi <= 50) return "#10B981";
  if (aqi <= 100) return "#84CC16";
  if (aqi <= 200) return "#F59E0B";
  if (aqi <= 300) return "#F97316";
  if (aqi <= 400) return "#EF4444";
  return "#7F1D1D";
}

/* ==========================================================================
   SLIDERS & AUTO-FILL DEFAULTS
   ========================================================================== */
function autoFillCityDefaults() {
  const cityName = document.getElementById("input-city").value;
  const cityObj = citiesData.find(c => c.name === cityName);
  if (!cityObj || !cityObj.defaults) return;

  const d = cityObj.defaults;
  setSlider("pm25", d["PM2.5"] || 50);
  setSlider("pm10", d["PM10"] || 90);
  setSlider("no2", d["NO2"] || 25);
  setSlider("so2", d["SO2"] || 10);
  setSlider("co", d["CO"] || 0.8);
  setSlider("o3", d["O3"] || 35);
}

function setSlider(name, val) {
  const el = document.getElementById(`input-${name}`);
  const display = document.getElementById(`val-${name}`);
  if (el) el.value = val;
  if (display) display.innerText = Number(val).toFixed(name === "co" ? 1 : 0);
}

function updateSliderDisplay(name) {
  const el = document.getElementById(`input-${name}`);
  const display = document.getElementById(`val-${name}`);
  if (el && display) {
    display.innerText = Number(el.value).toFixed(name === "co" ? 1 : 0);
  }
}

/* ==========================================================================
   PREDICTION & VISUALIZATION RENDERING
   ========================================================================== */
async function handlePredict(e) {
  if (e) e.preventDefault();

  const city = document.getElementById("input-city").value;
  const date = document.getElementById("input-date").value;
  const pm25 = parseFloat(document.getElementById("input-pm25").value);
  const pm10 = parseFloat(document.getElementById("input-pm10").value);
  const no2 = parseFloat(document.getElementById("input-no2").value);
  const so2 = parseFloat(document.getElementById("input-so2").value);
  const co = parseFloat(document.getElementById("input-co").value);
  const o3 = parseFloat(document.getElementById("input-o3").value);

  const payload = { city, date, pm25, pm10, no2, so2, co, o3 };

  const btn = document.getElementById("btn-predict");
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin mr-2"></i>Computing Forecast...`;
  }

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error(`Prediction error: ${res.status}`);

    const data = await res.json();
    currentPrediction = data;
    renderPredictionResult(data);

    // Render suitable charts
    renderForecastComparisonChart();
    renderPollutantLimitsChart();

  } catch (err) {
    console.error("Predict failed:", err);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles mr-2"></i>Run Stacking Forecast`;
    }
  }
}

function renderPredictionResult(data) {
  const aqi = data.predicted_aqi;
  const adv = data.advisory;
  const city = data.city;
  const date = data.date;

  // AQI Value & Badge
  document.getElementById("res-aqi-value").innerText = aqi;
  document.getElementById("res-city-date").innerText = `${city} Forecast (${date})`;
  document.getElementById("res-summary-text").innerText = adv.summary;
  document.getElementById("res-range-text").innerText = `(${adv.range})`;

  // Category Pill
  const pill = document.getElementById("res-category-pill");
  pill.innerText = adv.category;
  pill.style.backgroundColor = adv.color;
  pill.style.color = ["Good", "Satisfactory", "Moderate"].includes(adv.category) ? "#020617" : "#FFFFFF";

  // Badge Circle glow
  const badgeCircle = document.getElementById("aqi-badge-circle");
  badgeCircle.style.borderColor = adv.color;
  badgeCircle.style.boxShadow = `0 0 25px ${adv.color}33`;
  document.getElementById("res-aqi-value").style.color = adv.color;

  // Health Advisories
  document.getElementById("adv-general").innerText = adv.general_population;
  document.getElementById("adv-sensitive").innerText = adv.sensitive_groups;
  document.getElementById("adv-workers").innerText = adv.outdoor_workers;

  // Actions List
  const actionsList = document.getElementById("adv-actions-list");
  actionsList.innerHTML = "";
  adv.actions.forEach(action => {
    const li = document.createElement("li");
    li.className = "flex items-center";
    li.innerHTML = `<i class="fa-solid fa-check mr-2 text-[10px]" style="color: ${adv.color}"></i><span>${action}</span>`;
    actionsList.appendChild(li);
  });
}

/* Visualization 1: Today's Forecast vs 7-Day & 30-Day Baselines */
function renderForecastComparisonChart() {
  const canvas = document.getElementById("chart-forecast-comparison");
  if (!canvas || !currentPrediction || !currentCitySummary) return;
  if (typeof Chart === "undefined") {
    console.warn("Chart.js not yet loaded");
    return;
  }

  const dark = isDarkMode();
  const textColor = dark ? "#cbd5e1" : "#334155";
  const gridColor = dark ? "#1e293b" : "#e2e8f0";

  const ctx = canvas.getContext("2d");
  if (chartForecastComparison) chartForecastComparison.destroy();

  const labels = ["Today (Predicted)", "Past 7-Day Avg", "Past 30-Day Avg"];
  const values = [
    currentPrediction.predicted_aqi,
    currentCitySummary.avg_7_days,
    currentCitySummary.avg_30_days
  ];
  const colors = [
    currentPrediction.advisory.color,
    "#06B6D4",
    "#64748B"
  ];

  chartForecastComparison = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "AQI Value",
        data: values,
        backgroundColor: colors,
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          grid: { color: gridColor },
          ticks: { color: textColor, font: { size: 10 } }
        },
        x: {
          grid: { display: false },
          ticks: { color: textColor, font: { size: 10, weight: "bold" } }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `AQI: ${ctx.raw}`
          }
        }
      }
    }
  });
}

/* Visualization 2: Pollutant Concentrations vs CPCB Safe Limits */
function renderPollutantLimitsChart() {
  const canvas = document.getElementById("chart-pollutant-limits");
  if (!canvas || !currentPrediction) return;
  if (typeof Chart === "undefined") {
    console.warn("Chart.js not yet loaded");
    return;
  }

  const dark = isDarkMode();
  const textColor = dark ? "#cbd5e1" : "#334155";
  const gridColor = dark ? "#1e293b" : "#e2e8f0";

  const ctx = canvas.getContext("2d");
  if (chartPollutantLimits) chartPollutantLimits.destroy();

  const p = currentPrediction.features_applied;
  const labels = ["PM2.5", "PM10", "NO2", "SO2", "O3"];
  const currentVals = [p.pm25, p.pm10, p.no2, p.so2, p.o3];
  const safeLimits = [CPCB_LIMITS.pm25, CPCB_LIMITS.pm10, CPCB_LIMITS.no2, CPCB_LIMITS.so2, CPCB_LIMITS.o3];

  chartPollutantLimits = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Current Level",
          data: currentVals,
          backgroundColor: currentVals.map((v, i) => v > safeLimits[i] ? "#EF4444" : "#10B981"),
          borderRadius: 4
        },
        {
          label: "CPCB Safe Limit",
          data: safeLimits,
          backgroundColor: dark ? "#475569" : "#cbd5e1",
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          grid: { color: gridColor },
          ticks: { color: textColor, font: { size: 10 } }
        },
        x: {
          grid: { display: false },
          ticks: { color: textColor, font: { size: 10, weight: "bold" } }
        }
      },
      plugins: {
        legend: {
          labels: { color: textColor, font: { size: 10 } }
        }
      }
    }
  });
}

/* ==========================================================================
   TAB 2: MODEL BENCHMARK LOGIC
   ========================================================================== */
async function loadModelBenchmarks() {
  try {
    const res = await fetch("/api/models/comparison");
    const data = await res.json();
    const metrics = data.metrics || {};

    const tbody = document.getElementById("benchmark-table-body");
    if (!tbody) return;
    tbody.innerHTML = "";

    const modelNames = Object.keys(metrics);
    const r2TestScores = [];
    const r2TrainScores = [];
    const maeScores = [];
    const rmseScores = [];

    modelNames.forEach(name => {
      const m = metrics[name];
      const isChampion = name === "Stacking Ensemble";

      r2TestScores.push(m.test_r2);
      r2TrainScores.push(m.train_r2);
      maeScores.push(m.test_mae);
      rmseScores.push(m.test_rmse);

      const tr = document.createElement("tr");
      tr.className = isChampion 
        ? "bg-cyan-500/10 dark:bg-cyan-950/30 font-bold border-l-4 border-cyan-500 text-slate-900 dark:text-white" 
        : "hover:bg-slate-100 dark:hover:bg-slate-800/40 text-slate-700 dark:text-slate-300";

      tr.innerHTML = `
        <td class="py-3 px-4 flex items-center">
          ${isChampion ? '<i class="fa-solid fa-crown text-amber-500 mr-2"></i>' : ''}
          <span class="${isChampion ? 'text-cyan-600 dark:text-cyan-400 font-extrabold' : ''}">${name}</span>
        </td>
        <td class="py-3 px-4 uppercase text-[10px] text-slate-500 dark:text-slate-400">${m.type.replace('_', ' ')}</td>
        <td class="py-3 px-4 font-bold ${isChampion ? 'text-emerald-600 dark:text-emerald-400 text-sm' : 'text-slate-900 dark:text-slate-200'}">${m.test_r2}</td>
        <td class="py-3 px-4 text-slate-500 dark:text-slate-400">${m.train_r2}</td>
        <td class="py-3 px-4 font-semibold text-cyan-600 dark:text-cyan-400">${m.overfit_gap_r2}</td>
        <td class="py-3 px-4 font-bold ${isChampion ? 'text-emerald-600 dark:text-emerald-400' : 'text-slate-800 dark:text-slate-300'}">${m.test_mae}</td>
        <td class="py-3 px-4 text-slate-500 dark:text-slate-400">${m.test_rmse}</td>
        <td class="py-3 px-4 text-slate-500 dark:text-slate-400">${m.cv_r2_mean} ±${m.cv_r2_std}</td>
      `;
      tbody.appendChild(tr);
    });

    renderBenchmarkCharts(modelNames, r2TestScores, r2TrainScores, maeScores, rmseScores);

  } catch (err) {
    console.error("Failed to load model benchmarks:", err);
  }
}

function renderBenchmarkCharts(names, testR2, trainR2, mae, rmse) {
  const dark = isDarkMode();
  const textColor = dark ? "#cbd5e1" : "#334155";
  const gridColor = dark ? "#1e293b" : "#e2e8f0";

  const ctxR2 = document.getElementById("chart-r2-comparison")?.getContext("2d");
  if (ctxR2) {
    if (chartR2) chartR2.destroy();
    chartR2 = new Chart(ctxR2, {
      type: "bar",
      data: {
        labels: names,
        datasets: [
          {
            label: "Test R² (Holdout)",
            data: testR2,
            backgroundColor: names.map(n => n === "Stacking Ensemble" ? "#10B981" : "#06B6D4"),
            borderRadius: 6
          },
          {
            label: "Train R²",
            data: trainR2,
            backgroundColor: dark ? "#475569" : "#94a3b8",
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { min: 0.85, max: 1.0, grid: { color: gridColor }, ticks: { color: textColor } },
          x: { grid: { display: false }, ticks: { color: textColor, font: { size: 10 } } }
        },
        plugins: {
          legend: { labels: { color: textColor, font: { size: 11 } } }
        }
      }
    });
  }

  const ctxErr = document.getElementById("chart-error-comparison")?.getContext("2d");
  if (ctxErr) {
    if (chartError) chartError.destroy();
    chartError = new Chart(ctxErr, {
      type: "bar",
      data: {
        labels: names,
        datasets: [
          {
            label: "Test MAE",
            data: mae,
            backgroundColor: names.map(n => n === "Stacking Ensemble" ? "#F59E0B" : "#F43F5E"),
            borderRadius: 6
          },
          {
            label: "Test RMSE",
            data: rmse,
            backgroundColor: dark ? "#64748b" : "#cbd5e1",
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { grid: { color: gridColor }, ticks: { color: textColor } },
          x: { grid: { display: false }, ticks: { color: textColor, font: { size: 10 } } }
        },
        plugins: {
          legend: { labels: { color: textColor, font: { size: 11 } } }
        }
      }
    });
  }
}

/* ==========================================================================
   TAB 3: HISTORICAL TRENDS LOGIC
   ========================================================================== */
async function loadCityHistoricalTrend() {
  const city = document.getElementById("trend-city-select")?.value || "Chennai";
  const limit = document.getElementById("trend-limit-select")?.value || 120;

  try {
    const res = await fetch(`/api/historical?city=${encodeURIComponent(city)}&limit=${limit}`);
    const data = await res.json();
    const history = data.history || [];

    const labels = history.map(h => h.date);
    const aqiVals = history.map(h => h.aqi);
    const pm25Vals = history.map(h => h.pm25);

    const dark = isDarkMode();
    const textColor = dark ? "#cbd5e1" : "#334155";
    const gridColor = dark ? "#1e293b" : "#e2e8f0";

    const canvas = document.getElementById("chart-historical-trend");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (chartHistorical) chartHistorical.destroy();

    chartHistorical = new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Daily AQI",
            data: aqiVals,
            borderColor: "#06b6d4",
            backgroundColor: dark ? "rgba(6, 182, 212, 0.1)" : "rgba(6, 182, 212, 0.05)",
            borderWidth: 2,
            pointRadius: 1.5,
            tension: 0.2,
            fill: true
          },
          {
            label: "PM2.5 (µg/m³)",
            data: pm25Vals,
            borderColor: "#f59e0b",
            borderWidth: 1.5,
            pointRadius: 0,
            borderDash: [4, 4],
            tension: 0.2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { grid: { color: gridColor }, ticks: { color: textColor } },
          x: { grid: { display: false }, ticks: { color: textColor, maxTicksLimit: 12 } }
        },
        plugins: {
          legend: { labels: { color: textColor } }
        }
      }
    });

  } catch (err) {
    console.error("Historical trend load failed:", err);
  }
}

/* ==========================================================================
   TAB 4: ERROR ANALYSIS LOGIC
   ========================================================================== */
async function loadErrorAnalysis() {
  try {
    const res = await fetch("/api/error-analysis");
    const data = await res.json();

    const catAccEl = document.getElementById("err-cat-acc");
    const overallMaeEl = document.getElementById("err-overall-mae");
    if (catAccEl) catAccEl.innerText = `${data.overall.category_classification_accuracy_percent}%`;
    if (overallMaeEl) overallMaeEl.innerText = `${data.overall.mae}`;

    const catTable = document.getElementById("category-error-table");
    if (!catTable) return;
    catTable.innerHTML = "";

    const cats = data.performance_by_category || {};
    Object.keys(cats).forEach(catName => {
      const c = cats[catName];
      const tr = document.createElement("tr");
      tr.className = "hover:bg-slate-100 dark:hover:bg-slate-800/40 text-slate-700 dark:text-slate-300";
      tr.innerHTML = `
        <td class="py-3 px-4 font-bold text-slate-900 dark:text-white">${catName}</td>
        <td class="py-3 px-4">${c.percentage_of_test}% (${c.sample_count})</td>
        <td class="py-3 px-4">${c.mean_true_aqi}</td>
        <td class="py-3 px-4 font-bold text-emerald-600 dark:text-emerald-400">${c.mae}</td>
        <td class="py-3 px-4 text-slate-500 dark:text-slate-400">${c.rmse}</td>
        <td class="py-3 px-4 ${c.mean_bias < 0 ? 'text-amber-600 dark:text-amber-400' : 'text-cyan-600 dark:text-cyan-400'}">${c.mean_bias > 0 ? '+' : ''}${c.mean_bias}</td>
        <td class="py-3 px-4 text-xs text-slate-500 dark:text-slate-400">${c.behavior}</td>
      `;
      catTable.appendChild(tr);
    });

  } catch (err) {
    console.error("Error analysis fetch error:", err);
  }
}

// Expose all interactive functions explicitly on window for HTML event handlers
window.switchTab = switchTab;
window.toggleTheme = toggleTheme;
window.selectQuickCity = selectQuickCity;
window.handlePredict = handlePredict;
window.autoFillCityDefaults = autoFillCityDefaults;
window.onCityChanged = onCityChanged;
window.loadCityHistoricalTrend = loadCityHistoricalTrend;
window.updateSliderDisplay = updateSliderDisplay;
