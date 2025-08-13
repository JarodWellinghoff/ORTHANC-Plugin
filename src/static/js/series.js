// Add unified CHO analysis button to the series page
$("#series").live("pagebeforeshow", function () {
  // Create main CHO analysis button
  var choButtonContainer = $("<div>").css({
    "margin-bottom": "20px",
    "text-align": "center",
  });
  let results = null;
  var choAnalysisButton = $("<a>")
    .attr("id", "cho-analysis-btn")
    .attr("data-role", "button")
    .attr("href", "#")
    .attr("data-icon", "gear")
    .attr("data-theme", "b")
    .text("CHO Analysis")
    .button();

  choButtonContainer.append(choAnalysisButton);
  choButtonContainer.insertBefore(
    $("#series-delete").parent().parent().parent().parent()
  );

  // Create the CHO analysis modal
  var choModal = $(`
    <div id="cho-modal" class="cho-modal" style="display: none;">
      <div class="cho-modal-content">
        <div class="cho-modal-header">
          <h2>CHO Analysis Configuration</h2>
          <button class="cho-close">&times;</button>
        </div>
        <div class="cho-modal-body">
          <!-- Test Selection -->
          <div id="cho-test-selection" class="cho-section">
            <h3>Analysis Type</h3>
            <div class="cho-radio-group">
              <label>
                <input type="radio" name="cho-test-type" value="global" checked>
                <span>Global Noise Analysis</span>
                <small>Fast analysis for noise characteristics</small>
              </label>
              <label>
                <input type="radio" name="cho-test-type" value="full">
                <span>Full CHO Analysis</span>
                <small>Comprehensive analysis with lesion detectability</small>
              </label>
            </div>
          </div>

          <!-- Common Parameters -->
          <div id="cho-common-params" class="cho-section">
            <h3>Common Parameters</h3>
            <div class="cho-param-grid">
              <div class="cho-param">
                <label for="cho-resamples">Number of Resamples:</label>
                <input type="number" id="cho-resamples" value="500" min="100" max="2000" step="100">
              </div>
              <div class="cho-param">
                <label for="cho-internal-noise">Internal Noise:</label>
                <input type="number" id="cho-internal-noise" value="2.25" min="0" max="10" step="0.25">
              </div>
              <div class="cho-param">
                <label for="cho-resampling-method">Resampling Method:</label>
                <select id="cho-resampling-method">
                  <option value="Bootstrap">Bootstrap</option>
                  <option value="Shuffle">Shuffle</option>
                </select>
              </div>
            </div>
          </div>

          <!-- Global Noise Specific Parameters -->
          <div id="cho-global-params" class="cho-section cho-params-global">
            <h3>Global Noise Parameters</h3>
            <div class="cho-param-grid">
              <div class="cho-param">
                <label for="cho-roi-size">ROI Size (mm):</label>
                <input type="number" id="cho-roi-size" value="6" min="3" max="15" step="1">
              </div>
              <div class="cho-param">
                <label for="cho-threshold-low">Lower Threshold (HU):</label>
                <input type="number" id="cho-threshold-low" value="0" min="-100" max="100" step="10">
              </div>
              <div class="cho-param">
                <label for="cho-threshold-high">Upper Threshold (HU):</label>
                <input type="number" id="cho-threshold-high" value="150" min="100" max="300" step="10">
              </div>
            </div>
          </div>

          <!-- Full Analysis Specific Parameters -->
          <div id="cho-full-params" class="cho-section cho-params-full" style="display: none;">
            <h3>Full Analysis Parameters</h3>
            <div class="cho-param-grid">
              <div class="cho-param">
                <label for="cho-window-length">Window Length (cm):</label>
                <input type="number" id="cho-window-length" value="15" min="10" max="25" step="2.5">
              </div>
              <div class="cho-param">
                <label for="cho-step-size">Step Size (cm):</label>
                <input type="number" id="cho-step-size" value="5" min="2.5" max="10" step="2.5">
              </div>
              <div class="cho-param">
                <label for="cho-channel-type">Channel Type:</label>
                <select id="cho-channel-type">
                  <option value="Gabor">Gabor</option>
                  <option value="Laguerre-Gauss">Laguerre-Gauss</option>
                </select>
              </div>
              <div class="cho-param">
                <label for="cho-lesion-set">Lesion Set:</label>
                <select id="cho-lesion-set">
                  <option value="standard">Standard (-30, -30, -10, -30, -50 HU)</option>
                  <option value="low-contrast">Low Contrast (-10, -15, -20 HU)</option>
                  <option value="high-contrast">High Contrast (-50, -75, -100 HU)</option>
                </select>
              </div>
            </div>
          </div>

          <!-- Progress Section -->
          <div id="cho-progress-section" class="cho-section" style="display: none;">
            <h3>Analysis Progress</h3>
            <div class="cho-progress-container">
              <div class="cho-progress-bar">
                <div id="cho-progress-fill" class="cho-progress-fill"></div>
              </div>
              <div id="cho-progress-text" class="cho-progress-text">0%</div>
            </div>
            <div id="cho-progress-message" class="cho-progress-message">Initializing...</div>
            <div id="cho-progress-stage" class="cho-progress-stage">
              <div class="cho-stage-indicators">
                <div class="cho-stage-indicator" data-stage="initialization">Init</div>
                <div class="cho-stage-indicator" data-stage="loading">Load</div>
                <div class="cho-stage-indicator" data-stage="preprocessing">Prep</div>
                <div class="cho-stage-indicator" data-stage="analysis">Analysis</div>
                <div class="cho-stage-indicator" data-stage="finalizing">Final</div>
              </div>
            </div>
          </div>

          <!-- Results Section -->
          <div id="cho-results-section" class="cho-section" style="display: none;">
            <h3>Analysis Results</h3>
            <div id="cho-results-content" class="cho-results-content">
              <!-- Results will be populated here -->
            </div>
            <div class="cho-results-actions">
              <button id="cho-save-results" class="cho-btn cho-btn-primary">Save Results to Database</button>
              <button id="cho-discard-results" class="cho-btn cho-btn-secondary">Discard Results</button>
            </div>
          </div>
        </div>

        <div class="cho-modal-footer">
          <button id="cho-start-analysis" class="cho-btn cho-btn-primary">Start Analysis</button>
          <button id="cho-cancel" class="cho-btn cho-btn-secondary">Cancel</button>
        </div>
      </div>
    </div>
  `);

  // Add modal to page
  $("body").append(choModal);

  var modalStyles = $(`
    <link
      rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" />
    <link rel="stylesheet" href="/static/css/dashboard.css" />
  `);

  $("head").append(modalStyles);

  // Modal functionality
  var currentSeriesUuid = null;
  var currentCalculationStatus = null;
  var progressInterval = null;

  // Button click handler
  choAnalysisButton.click(function () {
    if ($.mobile.pageData) {
      currentSeriesUuid = $.mobile.pageData.uuid;
      openChoModal();
    }
  });

  // Modal open/close functions
  function openChoModal() {
    $("#cho-modal").show().addClass("show");
    resetModal();
    checkExistingCalculation();
  }

  function closeChoModal() {
    $("#cho-modal").hide();
    if (progressInterval) {
      clearInterval(progressInterval);
      progressInterval = null;
    }
  }

  function resetModal() {
    // Reset to initial state
    $("#cho-test-selection, #cho-common-params, #cho-global-params").show();
    $("#cho-progress-section, #cho-results-section").hide();
    $("#cho-start-analysis").show().prop("disabled", false);
    updateButtonState("idle");
  }

  function updateButtonState(state) {
    var button = $("#cho-analysis-btn");
    switch (state) {
      case "idle":
        button
          .find(".ui-icon")
          .removeClass("ui-icon-refresh ui-icon-check")
          .addClass("ui-icon-gear");
        button.find(".ui-btn-text").text("CHO Analysis");
        break;
      case "running":
        button
          .find(".ui-icon")
          .removeClass("ui-icon-gear ui-icon-check")
          .addClass("ui-icon-refresh");
        button.find(".ui-btn-text").text("Analysis Running...");
        break;
      case "results":
        button
          .find(".ui-icon")
          .removeClass("ui-icon-gear ui-icon-refresh")
          .addClass("ui-icon-check");
        button.find(".ui-btn-text").text("Results Available");
        break;
    }
  }

  // Test type change handler
  $('input[name="cho-test-type"]').change(function () {
    if ($(this).val() === "global") {
      $(".cho-params-global").show();
      $(".cho-params-full").hide();
    } else {
      $(".cho-params-global").hide();
      $(".cho-params-full").show();
    }
  });

  // Modal event handlers
  $(".cho-close, #cho-cancel").click(function () {
    closeChoModal();
  });

  $("#cho-start-analysis").click(function () {
    startAnalysis();
  });

  $("#cho-save-results").click(function () {
    saveResults();
  });

  $("#cho-discard-results").click(function () {
    discardResults();
  });

  // Main analysis function
  function startAnalysis() {
    var testType = $('input[name="cho-test-type"]:checked').val();
    var params = collectParameters();

    // Hide configuration sections
    $(
      "#cho-test-selection, #cho-common-params, #cho-global-params, #cho-full-params"
    ).hide();
    $("#cho-progress-section").show();
    $("#cho-start-analysis").hide();

    updateButtonState("running");
    $.ajax({
      url: "/cho-analysis-modal",
      type: "POST",
      contentType: "application/json",
      data: JSON.stringify({
        series_uuid: currentSeriesUuid,
        ...params,
      }),
      success: function (response) {
        console.log("Analysis started:", response);
        startProgressPolling();
      },
      error: function (xhr, status, error) {
        console.error("Failed to start analysis:", error);
        showError("Failed to start analysis: " + error);
      },
    });
  }

  function collectParameters() {
    return {
      testType: $('input[name="cho-test-type"]:checked').val(),
      resamples: parseInt($("#cho-resamples").val()),
      internalNoise: parseFloat($("#cho-internal-noise").val()),
      resamplingMethod: $("#cho-resampling-method").val(),
      roiSize: parseInt($("#cho-roi-size").val()),
      thresholdLow: parseInt($("#cho-threshold-low").val()),
      thresholdHigh: parseInt($("#cho-threshold-high").val()),
      windowLength: parseFloat($("#cho-window-length").val()),
      stepSize: parseFloat($("#cho-step-size").val()),
      channelType: $("#cho-channel-type").val(),
      lesionSet: $("#cho-lesion-set").val(),
      saveResults: false,
    };
  }

  function startProgressPolling() {
    progressInterval = setInterval(function () {
      checkCalculationStatus();
    }, 1000);
  }

  function checkExistingCalculation() {
    $.get(`/cho-calculation-status?series_id=${currentSeriesUuid}&action=check`)
      .done(function (data) {
        if (data.status === "running") {
          // Show progress for existing calculation
          showProgressSection();
          updateProgress(data);
          startProgressPolling();
          updateButtonState("running");
        } else if (data.status === "completed") {
          // Show results for completed calculation
          showResultsSection(data.results);
          updateButtonState("results");
        }
      })
      .fail(function () {
        // No existing calculation
        resetModal();
      });
  }

  function checkCalculationStatus() {
    $.get(`/cho-calculation-status?series_id=${currentSeriesUuid}&action=check`)
      .done(function (data) {
        updateProgress(data);

        if (data.status === "completed") {
          clearInterval(progressInterval);
          progressInterval = null;
          results = data.results;
          showResultsSection(results);
          updateButtonState("results");
        } else if (data.status === "failed") {
          clearInterval(progressInterval);
          progressInterval = null;
          showError(data.error);
          updateButtonState("idle");
        }
      })
      .fail(function () {
        console.error("Failed to get calculation status");
      });
  }

  function updateProgress(data) {
    var progress = data.progress || 0;
    $("#cho-progress-fill").css("width", progress + "%");
    $("#cho-progress-text").text(progress + "%");
    $("#cho-progress-message").text(data.message || "Processing...");

    // Update stage indicators
    $(".cho-stage-indicator").removeClass("active completed");
    $(".cho-stage-indicator").each(function () {
      var stage = $(this).data("stage");
      if (stage === data.current_stage) {
        $(this).addClass("active");
      } else {
        // Mark as completed if it comes before current stage
        var stages = [
          "initialization",
          "loading",
          "preprocessing",
          "analysis",
          "finalizing",
        ];
        var currentIndex = stages.indexOf(data.current_stage);
        var thisIndex = stages.indexOf(stage);
        if (thisIndex < currentIndex) {
          $(this).addClass("completed");
        }
      }
    });
  }

  function showProgressSection() {
    $(
      "#cho-test-selection, #cho-common-params, #cho-global-params, #cho-full-params"
    ).hide();
    $("#cho-progress-section").show();
    $("#cho-start-analysis").hide();
  }

  function showResultsSection(results) {
    $("#cho-progress-section").hide();
    $("#cho-results-section").show();

    // Format and display results
    var resultsHtml = formatResults(results);
    $("#cho-results-content").html(resultsHtml);

    currentCalculationStatus = results;
  }

  function formatResults(results) {
    if (!results) return "<p>No results available</p>";

    var html = '<div class="cho-results-grid">';

    // Key metrics
    var metrics = [
      { label: "Processing Time", value: results.processing_time, unit: "s" },
      { label: "Average CTDI", value: results.ctdivol_avg, unit: "mGy" },
      { label: "SSDE", value: results.ssde, unit: "mGy" },
      {
        label: "Average Noise Level",
        value: results.average_noise_level,
        unit: "HU",
      },
    ];

    if (results.average_index_of_detectability) {
      metrics.push({
        label: "Mean Detectability Index",
        value: results.average_index_of_detectability,
        unit: "",
      });
    }

    if (results.peak_frequency) {
      metrics.push({
        label: "Peak Frequency",
        value: results.peak_frequency,
        unit: "mm⁻¹",
      });
    }

    metrics.forEach(function (metric) {
      if (metric.value !== null && metric.value !== undefined) {
        var value =
          typeof metric.value === "number"
            ? metric.value.toFixed(3)
            : metric.value;
        html += `
          <div class="cho-result-item">
            <div class="cho-result-label">${metric.label}</div>
            <div class="cho-result-value">${value} ${metric.unit}</div>
          </div>
        `;
      }
    });

    html += "</div>";

    // Add some styling for the results
    if (!$("#cho-results-styles").length) {
      $("head").append(`
        <style id="cho-results-styles">
          .cho-results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
          }
          
          .cho-result-item {
            padding: 15px;
            background: white;
            border-radius: 6px;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          }
          
          .cho-result-label {
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
          }
          
          .cho-result-value {
            font-size: 18px;
            font-weight: bold;
            color: #333;
          }
        </style>
      `);
    }

    return html;
  }

  function saveResults() {
    console.log(results);
    if (!results) return alert("Failed to save results.");
    $.ajax({
      url: "/cho-save-results",
      type: "POST",
      contentType: "application/json",
      data: JSON.stringify({
        patient: results.patient,
        study: results.study,
        scanner: results.scanner,
        series: results.series,
        ct: results.ct,
        converted_results: results.results,
      }),
    })
      .done(function () {
        alert("Results have been saved to the database successfully!");
      })
      .fail(function () {
        alert("Failed to save results.");
      });
    // Results are already saved by the backend, just close modal
    closeChoModal();
    updateButtonState("idle");
    results = null;
  }

  function discardResults() {
    if (confirm("Are you sure you want to discard these results?")) {
      // Could implement deletion here if needed
      closeChoModal();
      updateButtonState("idle");
    }
  }

  function showError(error) {
    $("#cho-progress-section").hide();
    $("#cho-results-section").show();
    $("#cho-results-content").html(`
      <div style="color: #e53e3e; text-align: center; padding: 20px;">
        <h4>Analysis Failed</h4>
        <p>${error || "Unknown error occurred"}</p>
      </div>
    `);
  }

  // Close modal when clicking outside
  $(window).click(function (event) {
    if (event.target.id === "cho-modal") {
      closeChoModal();
    }
  });
});
