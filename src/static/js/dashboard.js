// CHO Dashboard JavaScript - Updated for unified modal interface with Pagination

// Global variables
let currentView = "summary";
let summaryData = [];
let filterOptions = {};
let seriesIdToDelete = null;
let calculationTypeToDelete = null;
let currentChoSeriesUuid = null;
let choProgressInterval = null;
let results = null;

// Pagination state
let summaryPagination = {
  currentPage: 1,
  pageSize: 25,
  totalItems: 0,
  totalPages: 0,
};

// Initialize the dashboard
$(document).ready(function () {
  //   loadFilterOptions();
  loadData();
  setupEventListeners();
  setupChoModal();
});

function setupEventListeners() {
  // Filter change handlers
  $(
    "#patient-search, #institute-filter, #scanner-station-filter, #protocol-filter, #scanner-model-filter, #exam-date-from, #exam-date-to, #age-min, #age-max"
  ).on("change keyup", function () {
    updateAppliedFilters();
    summaryPagination.currentPage = 1;
    loadData();
  });

  // Modal handlers
  $(".close").click(function () {
    const modalId = $(this).closest(".modal").attr("id");
    hideModal(`#${modalId}`);
  });

  $(window).click(function (event) {
    if ($(event.target).hasClass("modal")) {
      hideModal(`#${event.target.id}`);
    }
  });

  // Confirmation modal handlers
  $("#confirm-delete").click(function () {
    if (seriesIdToDelete) {
      deleteResult(seriesIdToDelete, calculationTypeToDelete);
      seriesIdToDelete = null;
      calculationTypeToDelete = null;
    }
  });

  $("#cancel-delete").click(function () {
    $("#confirmationModal").hide();
    seriesIdToDelete = null;
    calculationTypeToDelete = null;
  });

  // Refresh data every 30 seconds
  //   setInterval(function () {
  //     loadData();
  //   }, 30000);
}

function setupChoModal() {
  // CHO Modal event handlers
  $("#cho-analysis-modal .cho-close, #cho-cancel").click(function () {
    closeChoModal();
  });

  $("#cho-start-analysis").click(function () {
    startChoAnalysis();
  });

  $("#cho-save-results").click(function () {
    saveChoResults();
  });

  $("#cho-discard-results").click(function () {
    discardChoResults();
  });

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

  // Close modal when clicking outside
  $(window).click(function (event) {
    if (event.target.id === "cho-analysis-modal") {
      closeChoModal();
    }
  });
}

function loadFilterOptions() {
  $.get("/cho-filter-options")
    .done(function (data) {
      filterOptions = data;
      populateFilterDropdowns();
      setDateRangeLimits();
    })
    .fail(function (xhr, status, error) {
      console.error("Error loading filter options:", error);
    });
}

function populateFilterDropdowns() {
  // Populate institute dropdown
  const instituteSelect = $("#institute-filter");
  instituteSelect.find("option:not(:first)").remove();
  filterOptions.institutes?.forEach((institute) => {
    instituteSelect.append(
      `<option value="${institute}">${institute}</option>`
    );
  });

  // Populate scanner station dropdown
  const scannerSelect = $("#scanner-station-filter");
  scannerSelect.find("option:not(:first)").remove();
  filterOptions.scanner_stations?.forEach((scanner) => {
    scannerSelect.append(`<option value="${scanner}">${scanner}</option>`);
  });

  // Populate protocol dropdown
  const protocolSelect = $("#protocol-filter");
  protocolSelect.find("option:not(:first)").remove();
  filterOptions.protocol_names?.forEach((protocol) => {
    protocolSelect.append(`<option value="${protocol}">${protocol}</option>`);
  });

  // Populate scanner model dropdown
  const modelSelect = $("#scanner-model-filter");
  modelSelect.find("option:not(:first)").remove();
  filterOptions.scanner_models?.forEach((model) => {
    modelSelect.append(`<option value="${model}">${model}</option>`);
  });
}

function setDateRangeLimits() {
  if (filterOptions.date_range?.min && filterOptions.date_range?.max) {
    const minDate = filterOptions.date_range.min.split("T")[0];
    const maxDate = filterOptions.date_range.max.split("T")[0];

    $("#exam-date-from, #exam-date-to")
      .attr("min", minDate)
      .attr("max", maxDate);
  }

  if (
    filterOptions.age_range?.min !== undefined &&
    filterOptions.age_range?.max !== undefined
  ) {
    $("#age-min")
      .attr("min", filterOptions.age_range.min)
      .attr("max", filterOptions.age_range.max);
    $("#age-max")
      .attr("min", filterOptions.age_range.min)
      .attr("max", filterOptions.age_range.max);
  }
}

function toggleAdvancedFilters() {
  const content = $("#filter-content");
  const toggle = $("#filter-toggle");

  if (content.hasClass("expanded")) {
    content.removeClass("expanded").slideUp();
    toggle.removeClass("expanded");
  } else {
    content.addClass("expanded").slideDown();
    toggle.addClass("expanded");
  }
}

function updateAppliedFilters() {
  const filtersContainer = $("#applied-filters");
  const tagsContainer = $("#filter-tags");

  // Clear existing tags except the label
  tagsContainer.find(".filter-tag").remove();

  const filters = [];

  // Check each filter for values
  const patientSearch = $("#patient-search").val();
  if (patientSearch)
    filters.push({
      type: "patient_search",
      value: patientSearch,
      label: `Patient: ${patientSearch}`,
    });

  const institute = $("#institute-filter").val();
  if (institute)
    filters.push({
      type: "institute",
      value: institute,
      label: `Institute: ${institute}`,
    });

  const scannerStation = $("#scanner-station-filter").val();
  if (scannerStation)
    filters.push({
      type: "scanner_station",
      value: scannerStation,
      label: `Station: ${scannerStation}`,
    });

  const protocol = $("#protocol-filter").val();
  if (protocol)
    filters.push({
      type: "protocol_name",
      value: protocol,
      label: `Protocol: ${protocol}`,
    });

  const scannerModel = $("#scanner-model-filter").val();
  if (scannerModel)
    filters.push({
      type: "scanner_model",
      value: scannerModel,
      label: `Model: ${scannerModel}`,
    });

  const dateFrom = $("#exam-date-from").val();
  const dateTo = $("#exam-date-to").val();
  if (dateFrom || dateTo) {
    const dateLabel = `Date: ${dateFrom || "any"} - ${dateTo || "any"}`;
    filters.push({
      type: "date_range",
      value: `${dateFrom}|${dateTo}`,
      label: dateLabel,
    });
  }

  const ageMin = $("#age-min").val();
  const ageMax = $("#age-max").val();
  if (ageMin || ageMax) {
    const ageLabel = `Age: ${ageMin || "0"} - ${ageMax || "∞"}`;
    filters.push({
      type: "age_range",
      value: `${ageMin}|${ageMax}`,
      label: ageLabel,
    });
  }

  // Show/hide filters container
  if (filters.length > 0) {
    filtersContainer.addClass("has-filters");
    filters.forEach((filter) => {
      const tag = $(`
        <span class="filter-tag">
          ${filter.label}
          <span class="remove" onclick="removeFilter('${filter.type}', '${filter.value}')">&times;</span>
        </span>
      `);
      tagsContainer.append(tag);
    });
  } else {
    filtersContainer.removeClass("has-filters");
  }
}

function removeFilter(type, value) {
  switch (type) {
    case "patient_search":
      $("#patient-search").val("");
      break;
    case "institute":
      $("#institute-filter").val("");
      break;
    case "scanner_station":
      $("#scanner-station-filter").val("");
      break;
    case "protocol_name":
      $("#protocol-filter").val("");
      break;
    case "scanner_model":
      $("#scanner-model-filter").val("");
      break;
    case "date_range":
      $("#exam-date-from, #exam-date-to").val("");
      break;
    case "age_range":
      $("#age-min, #age-max").val("");
      break;
  }
  updateAppliedFilters();
  summaryPagination.currentPage = 1;
  loadData();
}

function clearAllFilters() {
  $(
    "#patient-search, #institute-filter, #scanner-station-filter, #protocol-filter, #scanner-model-filter, #exam-date-from, #exam-date-to, #age-min, #age-max"
  ).val("");
  updateAppliedFilters();
  summaryPagination.currentPage = 1;
  loadData();
}

function getFilterParams() {
  const params = {};

  const patientSearch = $("#patient-search").val();
  if (patientSearch) params.patient_search = patientSearch;

  const institute = $("#institute-filter").val();
  if (institute) params.institute = institute;

  const scannerStation = $("#scanner-station-filter").val();
  if (scannerStation) params.scanner_station = scannerStation;

  const protocol = $("#protocol-filter").val();
  if (protocol) params.protocol_name = protocol;

  const scannerModel = $("#scanner-model-filter").val();
  if (scannerModel) params.scanner_model = scannerModel;

  const dateFrom = $("#exam-date-from").val();
  if (dateFrom) params.exam_date_from = dateFrom;

  const dateTo = $("#exam-date-to").val();
  if (dateTo) params.exam_date_to = dateTo;

  const ageMin = $("#age-min").val();
  if (ageMin) params.patient_age_min = ageMin;

  const ageMax = $("#age-max").val();
  if (ageMax) params.patient_age_max = ageMax;

  return params;
}

// Load data based on current view
function loadData() {
  loadFilterOptions();

  setStatus("Loading data...", "loading");

  const params = getFilterParams();

  // Add pagination parameters
  params.page = summaryPagination.currentPage;
  params.limit = summaryPagination.pageSize;

  $.get("/results-statistics")
    .done(function (response) {
      summaryData = response;
      updateSummaryStatistics(summaryData);
    })
    .fail(function (xhr, status, error) {
      setStatus(`Error: ${error}`, "error");
      console.error("Error loading series summary:", error);
    });

  $.get("/cho-results", params)
    .done(function (response) {
      // New paginated response
      summaryData = response.data || [];
      summaryPagination.totalItems = response.total || 0;
      summaryPagination.totalPages = response.pages || 1;
      summaryPagination.currentPage = response.page || 1;

      displaySeriesSummary(summaryData);
      updateSummaryPaginationControls();
      setStatus(
        `Loaded ${summaryData.length} series (Page ${summaryPagination.currentPage} of ${summaryPagination.totalPages})`,
        "success"
      );
    })
    .fail(function (xhr, status, error) {
      setStatus(`Error: ${error}`, "error");
      console.error("Error loading series summary:", error);
    });
}

// Pagination Functions
function changeSummaryPageSize() {
  summaryPagination.pageSize = parseInt($("#summary-page-size").val());
  summaryPagination.currentPage = 1;
  loadData();
}

function goToSummaryPage(page) {
  if (page === "last") {
    page = summaryPagination.totalPages;
  }
  summaryPagination.currentPage = Math.max(
    1,
    Math.min(page, summaryPagination.totalPages)
  );
  loadData();
}

function nextSummaryPage() {
  if (summaryPagination.currentPage < summaryPagination.totalPages) {
    summaryPagination.currentPage++;
    loadData();
  }
}

function prevSummaryPage() {
  if (summaryPagination.currentPage > 1) {
    summaryPagination.currentPage--;
    loadData();
  }
}

function updateSummaryPaginationControls() {
  const pagination = summaryPagination;

  // Update info text
  const start = Math.min(
    (pagination.currentPage - 1) * pagination.pageSize + 1,
    pagination.totalItems
  );
  const end = Math.min(
    pagination.currentPage * pagination.pageSize,
    pagination.totalItems
  );
  $("#summary-pagination-info").text(
    `Showing ${start} - ${end} of ${pagination.totalItems} entries`
  );

  // Update button states
  $("#summary-first-page, #summary-prev-page").prop(
    "disabled",
    pagination.currentPage <= 1
  );
  $("#summary-next-page, #summary-last-page").prop(
    "disabled",
    pagination.currentPage >= pagination.totalPages
  );

  // Generate page numbers
  renderSummaryPageNumbers();
}

function renderSummaryPageNumbers() {
  const container = $("#summary-page-numbers");
  container.empty();

  const { currentPage, totalPages } = summaryPagination;
  const maxVisible = 7; // Maximum number of page buttons to show

  if (totalPages <= maxVisible) {
    // Show all pages
    for (let i = 1; i <= totalPages; i++) {
      container.append(createPageButton(i, currentPage, "goToSummaryPage"));
    }
  } else {
    // Show pages with ellipsis
    let startPage = Math.max(1, currentPage - 2);
    let endPage = Math.min(totalPages, currentPage + 2);

    // Adjust range to always show 5 pages when possible
    if (endPage - startPage < 4) {
      if (startPage === 1) {
        endPage = Math.min(totalPages, startPage + 4);
      } else {
        startPage = Math.max(1, endPage - 4);
      }
    }

    // Always show first page
    if (startPage > 1) {
      container.append(createPageButton(1, currentPage, "goToSummaryPage"));
      if (startPage > 2) {
        container.append('<span class="pagination-ellipsis">...</span>');
      }
    }

    // Show range of pages
    for (let i = startPage; i <= endPage; i++) {
      container.append(createPageButton(i, currentPage, "goToSummaryPage"));
    }

    // Always show last page
    if (endPage < totalPages) {
      if (endPage < totalPages - 1) {
        container.append('<span class="pagination-ellipsis">...</span>');
      }
      container.append(
        createPageButton(totalPages, currentPage, "goToSummaryPage")
      );
    }
  }
}

function createPageButton(pageNum, currentPage, clickHandler) {
  const isActive = pageNum === currentPage;
  const activeClass = isActive ? " active" : "";
  return `<button class="pagination-btn${activeClass}" onclick="${clickHandler}(${pageNum})">${pageNum}</button>`;
}

// Load series analysis summary with pagination
// function loadSeriesSummary() {
//   setStatus("Loading data...", "loading");

//   const params = getFilterParams();

//   // Add pagination parameters
//   params.page = summaryPagination.currentPage;
//   params.limit = summaryPagination.pageSize;

//   $.get("/cho-results", params)
//     .done(function (response) {
//       // Handle both array response (legacy) and object response (with pagination)
//       if (Array.isArray(response)) {
//         // Legacy response - treat as if all data is on one page
//         summaryData = response;
//         summaryPagination.totalItems = response.length;
//         summaryPagination.totalPages = 1;
//         summaryPagination.currentPage = 1;
//       } else {
//         // New paginated response
//         summaryData = response.data || [];
//         summaryPagination.totalItems = response.total || 0;
//         summaryPagination.totalPages = response.pages || 1;
//         summaryPagination.currentPage = response.page || 1;
//       }

//       displaySeriesSummary(summaryData);
//       updateSummaryStatistics(summaryData);
//       updateSummaryPaginationControls();
//       setStatus(
//         `Loaded ${summaryData.length} series (Page ${summaryPagination.currentPage} of ${summaryPagination.totalPages})`,
//         "success"
//       );
//     })
//     .fail(function (xhr, status, error) {
//       setStatus(`Error: ${error}`, "error");
//       console.error("Error loading series summary:", error);
//     });
// }

// Display series summary - updated for unified CHO modal
function displaySeriesSummary(data) {
  const tbody = $("#summary-body");
  tbody.empty();

  // Get series UUIDs from Orthanc to check availability
  fetch("/series/")
    .then((response) => response.json())
    .then((orthancSeries) => {
      console.log("Orthanc series data:", orthancSeries);

      if (data.length === 0) {
        tbody.append(
          '<tr><td colspan="8" class="loading">No series found matching the filters</td></tr>'
        );
        return;
      }

      data.forEach(function (item) {
        const row = $("<tr>").click(function () {
          showSeriesDetails(item.series_id);
        });

        row.append($("<td>").text(item.patient_name || "N/A"));
        // row.append($("<td>").text(item.patient_id || "N/A"));
        row.append($("<td>").text(item.institution_name || "N/A"));
        row.append($("<td>").text(item.scanner_model || "N/A"));
        row.append($("<td>").text(item.station_name || "N/A"));
        row.append($("<td>").text(item.protocol_name || "N/A"));

        const disabled = !orthancSeries.includes(item.series_uuid);
        console.log(
          `Series UUID ${item.series_uuid} in DICOM database: ${!disabled}`
        );

        // Analysis status
        const statusCell = $("<td>");
        const statusSpan = $("<span>")
          .addClass(`analysis-status ${getStatusClass(item.test_status)}`)
          .text(getStatusDisplayText(item.test_status));
        statusCell.append(statusSpan);
        row.append(statusCell);

        row.append($("<td>").text(formatDateTime(item.latest_analysis_date)));

        const actionsCell = $("<td>");
        const actionButtonContainer = $("<div class='action-btn-container'>");

        // Unified CHO Analysis button - replaces the separate buttons
        const choAnalysisBtn = $("<button class='action-btn btn-run'>")
          .html('<i class="fas fa-cogs"></i>')
          .attr("title", "CHO Analysis")
          .click(function (event) {
            if (!disabled) {
              event.stopPropagation();
              openChoModal(item.series_uuid, item);
            }
          });

        if (disabled) {
          choAnalysisBtn.prop("disabled", true).addClass("tooltip");
          choAnalysisBtn.append(
            $("<span class='tooltiptext'>").text(
              "Series not found in DICOM database"
            )
          );
        }

        const viewSeriesBtn = $(
          `<button class="action-btn btn-viewer ${disabled ? "tooltip" : ""}" ${
            disabled ? "disabled" : ""
          }>`
        )
          .html('<i class="fas fa-eye"></i>')
          .click(function (event) {
            if (!disabled) {
              event.stopPropagation();
              window.open(
                "/ohif/viewer?StudyInstanceUIDs=" + item.study_id,
                "_blank"
              );
            }
          });

        const deleteBtn = $("<button class='action-btn btn-delete'>")
          .html('<i class="fas fa-trash"></i>')
          .click(function (event) {
            event.stopPropagation();
            confirmDelete(item.series_id, null, item.patient_name);
          });

        // Add tooltips for disabled buttons
        if (disabled) {
          [viewSeriesBtn].forEach((btn) => {
            btn.append(
              $("<span class='tooltiptext'>").text(
                "Series not found in DICOM database"
              )
            );
          });
        }

        actionButtonContainer.append(choAnalysisBtn, viewSeriesBtn, deleteBtn);
        actionsCell.append(actionButtonContainer);
        row.append(actionsCell);
        tbody.append(row);
      });
    })
    .catch((error) => {
      console.error("Error fetching Orthanc series:", error);
      tbody.append(
        '<tr><td colspan="8" class="loading">Error loading series data</td></tr>'
      );
    });
}

// Update summary statistics
function updateSummaryStatistics(data) {
  if (data.length === 0) {
    $("#total-series, #complete-series, #partial-series, #pending-series").text(
      0
    );
    return;
  }

  // For paginated data, we need to get total stats from the server response
  // For now, we'll calculate based on the current page data
  const totalSeries = data.total_results_count;
  const completeSeries = data.detectability_count;
  const pendingSeries = data.error_count;
  const partialSeries = data.global_noise_count;

  $("#total-series").text(totalSeries);
  $("#complete-series").text(completeSeries);
  $("#partial-series").text(partialSeries);
  $("#pending-series").text(pendingSeries);
}

// Helper functions updated for new schema
function getStatusDisplayText(test_status) {
  switch (test_status) {
    case "full":
      return "Detectability";
    case "partial":
      return "Global Noise";
    case "error":
      return "Error";
    default:
      return "Unknown Status";
  }
}

function getStatusClass(test_status) {
  switch (test_status) {
    case "full":
      return "status-completed";
    case "partial":
      return "status-partial";
    case "error":
      return "status-error";
    default:
      return "status-none";
  }
}

// CHO Modal Functions
function openChoModal(seriesUuid, seriesData) {
  currentChoSeriesUuid = seriesUuid;
  resetChoModal();
  checkExistingChoCalculation();
  showModal("#cho-analysis-modal");
}

function closeChoModal() {
  hideModal("#cho-analysis-modal");
  if (choProgressInterval) {
    clearInterval(choProgressInterval);
    choProgressInterval = null;
  }
}

function resetChoModal() {
  // Reset to initial state
  $("#cho-test-selection, #cho-common-params, #cho-global-params").show();
  $("#cho-progress-section, #cho-results-section").hide();
  $("#cho-start-analysis").show().prop("disabled", false);
  $('input[name="cho-test-type"][value="global"]').prop("checked", true);
  $(".cho-params-global").show();
  $(".cho-params-full").hide();
  $(".cho-results-actions").hide();
}

function startChoAnalysis() {
  const testType = $('input[name="cho-test-type"]:checked').val();
  const params = collectChoParameters();

  // Hide configuration sections
  $(
    "#cho-test-selection, #cho-common-params, #cho-global-params, #cho-full-params"
  ).hide();
  $("#cho-progress-section").show();
  $("#cho-start-analysis").hide();
  $(".cho-results-actions").hide();

  // Send request to start analysis
  $.ajax({
    url: "/cho-analysis-modal",
    method: "POST",
    contentType: "application/json",
    data: JSON.stringify({
      series_uuid: currentChoSeriesUuid,
      ...params,
    }),
    success: function (response) {
      console.log("Analysis started:", response);
      startChoProgressPolling();
    },
    error: function (xhr, status, error) {
      console.error("Failed to start analysis:", error);
      showChoError("Failed to start analysis: " + error);
    },
  });
}

function collectChoParameters() {
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

function startChoProgressPolling() {
  choProgressInterval = setInterval(function () {
    checkChoCalculationStatus();
  }, 1000);
}

function checkExistingChoCalculation() {
  $.get(
    `/cho-calculation-status?series_id=${currentChoSeriesUuid}&action=check`
  )
    .done(function (data) {
      if (data.status === "running") {
        showChoProgressSection();
        updateChoProgress(data);
        startChoProgressPolling();
      } else if (data.status === "completed") {
        showChoResultsSection(data.results);
        $(".cho-results-actions").show();
      }
    })
    .fail(function () {
      // No existing calculation
      resetChoModal();
    });
}

function checkChoCalculationStatus() {
  $.get(
    `/cho-calculation-status?series_id=${currentChoSeriesUuid}&action=check`
  )
    .done(function (data) {
      updateChoProgress(data);

      if (data.status === "completed") {
        clearInterval(choProgressInterval);
        choProgressInterval = null;
        results = data.results;
        showChoResultsSection(results);
        $(".cho-results-actions").show();
      } else if (data.status === "failed") {
        clearInterval(choProgressInterval);
        choProgressInterval = null;
        showChoError(data.error);
        $(".cho-results-actions").hide();
      }
    })
    .fail(function () {
      console.error("Failed to get calculation status");
    });
}

function updateChoProgress(data) {
  const progress = data.progress || 0;
  $("#cho-progress-fill").css("width", progress + "%");
  $("#cho-progress-text").text(progress + "%");
  $("#cho-progress-message").text(data.message || "Processing...");

  // Update stage indicators
  $(".cho-stage-indicator").removeClass("active completed");
  $(".cho-stage-indicator").each(function () {
    const stage = $(this).data("stage");
    if (stage === data.current_stage) {
      $(this).addClass("active");
    } else {
      const stages = [
        "initialization",
        "loading",
        "preprocessing",
        "analysis",
        "finalizing",
      ];
      const currentIndex = stages.indexOf(data.current_stage);
      const thisIndex = stages.indexOf(stage);
      if (thisIndex < currentIndex) {
        $(this).addClass("completed");
      }
    }
  });
}

function showChoProgressSection() {
  $(
    "#cho-test-selection, #cho-common-params, #cho-global-params, #cho-full-params"
  ).hide();
  $("#cho-progress-section").show();
  $("#cho-start-analysis").hide();
}

function showChoResultsSection(results) {
  console.log("Displaying CHO results:", results);
  $("#cho-progress-section").hide();
  $("#cho-results-section").show();
  createAndDisplayPlots(results, "results");
  createMetadataTable(results, "results");
  $("#details-content-results").text(JSON.stringify(results, null, 2));

  //   const resultsHtml = formatChoResults(results);
  //   $("#cho-results-content").html(resultsHtml);
}

function formatChoResults(results) {
  if (!results) return "<p>No results available</p>";

  let html = '<div class="cho-results-grid">';

  const metrics = [
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
      unit: "cm<sup>-1</sup>",
    });
  }

  metrics.forEach(function (metric) {
    if (metric.value !== null && metric.value !== undefined) {
      const value =
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
  return html;
}

function saveChoResults() {
  if (!results) return alert("Failed to save results.");

  $.ajax({
    url: "/cho-save-results",
    method: "POST",
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
  results = null;
  closeChoModal();
  loadData(); // Refresh the data
}

function discardChoResults() {
  if (confirm("Are you sure you want to discard these results?")) {
    $.get(
      `/cho-calculation-status?series_id=${currentChoSeriesUuid}&action=discard`
    )
      .done(function () {
        alert("Results have been discarded successfully!");
        closeChoModal();
      })
      .fail(function () {
        alert("Failed to discard results.");
      });
  }
}

function showChoError(error) {
  $("#cho-progress-section").hide();
  $("#cho-results-section").show();
  $("#cho-results-content").html(`
    <div style="color: #e53e3e; text-align: center; padding: 20px;">
      <h4>Analysis Failed</h4>
      <p>${error || "Unknown error occurred"}</p>
    </div>
  `);
}

// Helper function to check if image exists
function checkImageExists(imageUrl) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(true);
    img.onerror = () => resolve(false);
    img.src = imageUrl;
  });
}

function exportSeries(seriesId) {
  $.ajax({
    url: "/cho-export-results",
    method: "POST",
    contentType: "application/json",
    data: JSON.stringify({ series_ids: [seriesId] }),
    xhrFields: { responseType: "blob" },
    success: function (blob) {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${seriesId}-results.xls`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    },
    error: function (xhr, status, error) {
      console.error("Download failed:", error);
    },
  });
}

// Show series details with improved coronal image handling
function showSeriesDetails(seriesId) {
  setStatus("Loading series details...", "loading");
  $.get("/cho-results/" + seriesId)
    .done(function (data) {
      console.log("Series details data:", data);
      createAndDisplayPlots(data, "details");
      createMetadataTable(data, "details");
      $("#details-content-details").text(JSON.stringify(data, null, 2));
      showModal("#detailsModal");
      setStatus("", "");
      $("#export-individual-result").click(function () {
        exportSeries(seriesId);
      });
    })
    .fail(function (xhr, status, error) {
      setStatus(`Error loading details: ${error}`, "error");
    });
}

// Create and display plots with improved coronal image alignment
async function createAndDisplayPlots(data, modal) {
  const series_data = data.series || data;
  const results_data = data.results || data;
  const series_instance_uid = series_data.series_instance_uid;
  const scanLength = series_data.scan_length_cm;
  const coronalImageUrl = `/minio-images/${series_instance_uid}`;
  const imageExists = await checkImageExists(coronalImageUrl);
  let npsData = [];
  let npsLayout = {};
  let ctdiDwData = [];
  let ctdiDwLayout = {};
  let noiseChoData = [];
  let noiseChoLayout = {};
  const image = imageExists
    ? [
        {
          x: 0,
          y: 0,
          sizex: scanLength,
          sizey: 1,
          source: coronalImageUrl,
          xref: "x",
          yref: "paper",
          xanchor: "left",
          yanchor: "bottom",
          opacity: 0.8,
          layer: "below",
          sizing: "stretch",
        },
      ]
    : [];
  const spatial_frequency_data = results_data.spatial_frequency;
  const nps_data = results_data.nps;
  const location_data = results_data.location;
  const ctdi_vol_data = results_data.ctdivol;
  const dw_data = results_data.dw;
  const location_sparse_data = results_data.location_sparse;
  const noise_level_data = results_data.noise_level;
  const cho_detectability_data = results_data.cho_detectability;

  // Common image configuration for plots that align with anatomical position

  // NPS vs Spatial Frequency plot (no coronal image - not location-based)
  if (spatial_frequency_data && nps_data) {
    npsData = [
      {
        name: "NPS",
        x: spatial_frequency_data,
        y: nps_data,
        type: "scatter",
        mode: "lines",
        line: { width: 3, shape: "spline", color: "#667eea" },
      },
    ];

    npsLayout = {
      title: {
        text: "Noise Power Spectrum vs Spatial Frequency",
        font: { size: 16, color: "#2d3748" },
      },
      xaxis: {
        title: "Spatial Frequency (cm<sup>-1</sup>)",
        titlefont: { size: 14 },
        tickfont: { size: 12 },
      },
      yaxis: {
        title: "NPS",
        titlefont: { size: 14 },
        tickfont: { size: 12 },
      },
      plot_bgcolor: "white",
      paper_bgcolor: "white",
      margin: { l: 60, r: 40, t: 60, b: 60 },
      showlegend: false,
      font: { family: "Arial, sans-serif" },
    };
    Plotly.newPlot(
      `nps-v-spatial-frequency-plot-${modal}`,
      npsData,
      npsLayout,
      {
        responsive: true,
        displayModeBar: false,
      }
    );
    document.getElementById(
      `nps-v-spatial-frequency-plot-container-${modal}`
    ).style.display = "block";
  } else {
    // Remove the Plot
    Plotly.purge(`nps-v-spatial-frequency-plot-${modal}`);
    document.getElementById(
      `nps-v-spatial-frequency-plot-container-${modal}`
    ).style.display = "none";
  }

  // CTDI/Dw vs Location plot with aligned coronal image
  if (location_data && ctdi_vol_data && dw_data) {
    ctdiDwData = [
      {
        name: "CTDIvol",
        x: location_data,
        y: ctdi_vol_data,
        type: "scatter",
        mode: "lines+markers",
        line: { width: 3, shape: "spline", color: "#667eea" },
        marker: { size: 4 },
      },
      {
        name: "Water Equivalent Diameter",
        x: location_data,
        y: dw_data,
        type: "scatter",
        mode: "lines+markers",
        yaxis: "y2",
        line: { width: 3, shape: "spline", color: "#e53e3e" },
        marker: { size: 4 },
      },
    ];

    ctdiDwLayout = {
      xaxis: {
        title: "Location (cm)",
        titlefont: { size: 14 },
        tickfont: { size: 12 },
        range: [0, scanLength],
        showgrid: false,
      },
      yaxis: {
        title: "CTDIvol (mGy)",
        titlefont: { size: 14, color: "#667eea" },
        tickfont: { size: 12, color: "#667eea" },
        showgrid: false,
      },
      yaxis2: {
        title: "Water Equivalent Diameter (mm)",
        titlefont: { size: 14, color: "#e53e3e" },
        tickfont: { size: 12, color: "#e53e3e" },
        overlaying: "y",
        side: "right",
        showgrid: false,
      },
      plot_bgcolor: "white",
      paper_bgcolor: "white",
      margin: { l: 60, r: 60, t: 60, b: 60 },
      legend: {
        x: 0.02,
        y: 1.02,
        bgcolor: "rgba(255,255,255,0.8)",
        bordercolor: "#e2e8f0",
        borderwidth: 1,
        yanchor: "bottom",
      },
      images: image,
      font: { family: "Arial, sans-serif" },
    };
    Plotly.newPlot(
      `dw-ctdivol-v-location-plot-${modal}`,
      ctdiDwData,
      ctdiDwLayout,
      {
        responsive: true,
        displayModeBar: false,
      }
    );
    document.getElementById(
      `dw-ctdivol-v-location-plot-container-${modal}`
    ).style.display = "block";
  } else {
    // Remove the Plot
    Plotly.purge(`dw-ctdivol-v-location-plot-${modal}`);
    document.getElementById(
      `dw-ctdivol-v-location-plot-container-${modal}`
    ).style.display = "none";
  }

  // Noise Level and CHO Detectability vs Location with aligned coronal image
  if (location_sparse_data && noise_level_data && cho_detectability_data) {
    noiseChoData = [
      {
        name: "Local Noise Level",
        x: location_sparse_data,
        y: noise_level_data,
        type: "scatter",
        mode: "lines+markers",
        line: { width: 3, shape: "spline", color: "#667eea", dash: "dot" },
        marker: { size: 5 },
      },
      {
        name: "CHO Detectability Index",
        x: location_sparse_data,
        y: cho_detectability_data,
        yaxis: "y2",
        type: "scatter",
        mode: "lines+markers",
        line: { width: 3, shape: "spline", color: "#e53e3e", dash: "dot" },
        marker: { size: 5 },
      },
    ];

    noiseChoLayout = {
      xaxis: {
        title: "Anatomical Location (cm)",
        titlefont: { size: 14 },
        tickfont: { size: 12 },
        range: [0, scanLength],
        showgrid: false,
      },
      yaxis: {
        title: "Noise Level (HU)",
        titlefont: { size: 14, color: "#667eea" },
        tickfont: { size: 12, color: "#667eea" },
        showgrid: false,
      },
      yaxis2: {
        title: "CHO Detectability Index",
        titlefont: { size: 14, color: "#e53e3e" },
        tickfont: { size: 12, color: "#e53e3e" },
        overlaying: "y",
        side: "right",
        showgrid: false,
      },
      plot_bgcolor: "white",
      paper_bgcolor: "white",
      margin: { l: 60, r: 60, t: 60, b: 60 },
      legend: {
        x: 0.02,
        y: 1.02,
        bgcolor: "rgba(255,255,255,0.8)",
        bordercolor: "#e2e8f0",
        borderwidth: 1,
        yanchor: "bottom",
      },
      images: image,
      font: { family: "Arial, sans-serif" },
    };
    Plotly.newPlot(
      `noise-level-cho-detectability-v-location-sparse-plot-${modal}`,
      noiseChoData,
      noiseChoLayout,
      { responsive: true, displayModeBar: false }
    );

    document.getElementById(
      `noise-level-cho-detectability-v-location-sparse-plot-container-${modal}`
    ).style.display = "block";
  } else {
    // Remove the Plot and make the div hidden
    Plotly.purge(
      `noise-level-cho-detectability-v-location-sparse-plot-${modal}`
    );
    document.getElementById(
      `noise-level-cho-detectability-v-location-sparse-plot-container-${modal}`
    ).style.display = "none";
  }

  // Add image availability info to console
  if (!imageExists) {
    console.warn(
      `Coronal image not available for series ${series_instance_uid}`
    );
  }
}

// Create metadata table (keeping same function)
function createMetadataTable(data, modal) {
  const patient_data = data.patient || data;
  const study_data = data.study || data;
  const scanner_data = data.scanner || data;
  const series_data = data.series || data;
  const ct_data = data.ct || data;
  const results_data = data.results || data;

  const metadataContainer = document.getElementById(`metadata-table-${modal}`);

  const table = document.createElement("table");
  table.className = `metadata-table-${modal}`;

  const sections = [
    {
      title: "Patient Information",
      data: [
        {
          label: "Patient Name",
          value: patient_data.patient_name || "Anonymous",
          unit: "",
        },
        {
          label: "Patient ID",
          value: data.patient_id || "Anonymous",
          unit: "",
        },
        {
          label: "Patient Sex",
          value: patient_data.sex || "Anonymous",
          unit: "",
        },
        {
          label: "Birth Date",
          value: patient_data.birth_date || "Anonymous",
          unit: "",
        },
      ],
    },
    {
      title: "Results",
      data: [
        {
          label: "Average CTDI",
          value: formatNumber(results_data.ctdivol_avg),
          unit: "mGy",
        },
        { label: "SSDE", value: formatNumber(results_data.ssde), unit: "mGy" },
        {
          label: "DLP",
          value: formatNumber(results_data.dlp),
          unit: "mGy&#x2022;cm",
        },
        {
          label: "Average Dw",
          value: formatNumber(results_data.dw_avg),
          unit: "cm",
        },
        {
          label: "Average Noise Level",
          value: formatNumber(results_data.average_noise_level),
          unit: "HU",
        },
        {
          label: "Peak Frequency",
          value: formatNumber(results_data.peak_frequency, 3),
          unit: "cm<sup>-1</sup>",
        },
        {
          label: "Average Frequency",
          value: formatNumber(results_data.average_frequency, 3),
          unit: "cm<sup>-1</sup>",
        },
        {
          label: "10% Peak Frequency",
          value: formatNumber(results_data.percent_10_frequency, 3),
          unit: "cm<sup>-1</sup>",
        },
        {
          label: "DLP*SSDE",
          value: formatNumber(results_data.dlp_ssde, 3),
          unit: "",
        },
        {
          label: "Average Index of Detectability",
          value: formatNumber(results_data.average_index_of_detectability, 3),
          unit: "",
        },
      ],
    },
    {
      title: "Study Information",
      data: [
        {
          label: "Study ID",
          value: study_data.study_id || "N/A",
          unit: "",
        },
        {
          label: "Description",
          value: study_data.study_description || "N/A",
          unit: "",
        },
        {
          label: "Institution",
          value: study_data.institution_name || "N/A",
          unit: "",
        },
        {
          label: "Study Date",
          value: study_data.study_date || "N/A",
          unit: "",
        },
        {
          label: "Study Time",
          value: study_data.study_time || "N/A",
          unit: "",
        },
      ],
    },
    {
      title: "Series Information",
      data: [
        {
          label: "Description",
          value: series_data.series_description || "N/A",
          unit: "",
        },
        {
          label: "Body Part Examined",
          value: series_data.body_part_examined || "N/A",
          unit: "",
        },
        {
          label: "Convolution Kernel",
          value: series_data.convolution_kernel || "N/A",
          unit: "",
        },
        {
          label: "Image Count",
          value: formatNumber(series_data.image_count, 0),
          unit: "",
        },
        {
          label: "Modality",
          value: series_data.modality || "N/A",
          unit: "",
        },
        {
          label: "Patient Position",
          value: series_data.patient_position || "N/A",
          unit: "",
        },
        {
          label: "Pixel Spacing",
          value:
            `[${series_data.pixel_spacing_mm[0]}, ${series_data.pixel_spacing_mm[1]}]` ||
            "N/A",
          unit: "mm",
        },
        {
          label: "Protocol Name",
          value: series_data.protocol_name || "N/A",
          unit: "",
        },
        {
          label: "Rows",
          value: formatNumber(series_data.rows, 0),
          unit: "",
        },
        {
          label: "Scan Length",
          value: formatNumber(series_data.scan_length_cm, 3),
          unit: "cm",
        },
        {
          label: "Series Date",
          value: series_data.series_date || "N/A",
          unit: "",
        },
        {
          label: "Series Number",
          value: series_data.series_number || "N/A",
          unit: "",
        },
        {
          label: "Series Time",
          value: series_data.series_time || "N/A",
          unit: "",
        },
        {
          label: "Slice Thickness",
          value: formatNumber(series_data.slice_thickness_mm, 3),
          unit: "mm",
        },
        {
          label: "Columns",
          value: formatNumber(series_data.columns, 0),
          unit: "",
        },
      ],
    },
    {
      title: "Scanner Information",
      data: [
        {
          label: "Serial Number",
          value: scanner_data.serial_number || "N/A",
          unit: "",
        },
        {
          label: "Institution",
          value: scanner_data.institution_name || "N/A",
          unit: "",
        },
        {
          label: "Manufacturer",
          value: scanner_data.manufacturer || "N/A",
          unit: "",
        },
        {
          label: "Model Name",
          value: scanner_data.model_name || "N/A",
          unit: "",
        },
        {
          label: "Station Name",
          value: scanner_data.station_name || "N/A",
          unit: "",
        },
      ],
    },
    {
      title: "CT Technical Parameters",
      data: [
        {
          label: "Data Collection Diameter",
          value: formatNumber(ct_data.data_collection_diam_mm, 3),
          unit: "mm",
        },
        {
          label: "Distance: Source to Detector",
          value: formatNumber(ct_data.dist_src_detector_mm, 3),
          unit: "mm",
        },
        {
          label: "Distance: Source to Patient",
          value: formatNumber(ct_data.dist_src_patient_mm, 3),
          unit: "mm",
        },
        {
          label: "Exposure Type",
          value: data.exposure_modulation_type || "N/A",
          unit: "",
        },
        {
          label: "Exposure Time",
          value: formatNumber(ct_data.exposure_time_ms, 0),
          unit: "ms",
        },
        {
          label: "Filter Type",
          value: ct_data.filter_type || "N/A",
          unit: "",
        },
        {
          label: "Focal Spots",
          value: ct_data.focal_spots_mm || "N/A",
          unit: "mm",
        },
        {
          label: "Gantry Angle",
          value: formatNumber(ct_data.gantry_detector_tilt_deg, 3),
          unit: "deg",
        },
        {
          label: "Generator Power",
          value: formatNumber(ct_data.generator_power_kw, 3),
          unit: "kW",
        },
        {
          label: "kvp",
          value: formatNumber(ct_data.kvp, 3),
          unit: "kV",
        },
        {
          label: "Recon Diameter",
          value: formatNumber(ct_data.recon_diameter_mm, 3),
          unit: "mm",
        },
        {
          label: "Single Collimation Width",
          value: formatNumber(ct_data.single_collimation_width_mm, 3),
          unit: "mm",
        },
        {
          label: "Spiral Pitch Factor",
          value: formatNumber(ct_data.spiral_pitch_factor, 3),
          unit: "",
        },
        {
          label: "Table Feed Per Rotation",
          value: formatNumber(ct_data.table_feed_per_rot_mm, 3),
          unit: "mm",
        },
        {
          label: "Table Speed",
          value: formatNumber(ct_data.table_speed_mm_s, 3),
          unit: "mm/s",
        },
        {
          label: "Total Collimation Width",
          value: formatNumber(ct_data.total_collimation_width_mm, 3),
          unit: "mm",
        },
      ],
    },
  ];

  sections.forEach((section) => {
    const headerRow = table.insertRow();
    const headerCell = headerRow.insertCell();
    headerCell.colSpan = 2;
    headerCell.className = "section-header";
    headerCell.textContent = section.title;

    section.data.forEach((item) => {
      if (item.value !== "N/A") {
        const row = table.insertRow();
        const labelCell = row.insertCell();
        const valueCell = row.insertCell();

        labelCell.className = "metric-label";
        labelCell.textContent = item.label;

        valueCell.className = "metric-value";
        valueCell.innerHTML = `${item.value}<span class="metric-unit">${item.unit}</span>`;
      }
    });
  });

  metadataContainer.innerHTML = "";
  metadataContainer.appendChild(table);
}

// Confirm delete dialog
function confirmDelete(seriesId, calculationType, patientName) {
  seriesIdToDelete = seriesId;
  calculationTypeToDelete = calculationType;
  const typeText = calculationType
    ? calculationType === "global_noise"
      ? "Global Noise"
      : "Full Analysis"
    : "All Results";
  $("#delete-series-info-series").text(`Series: ${truncateText(seriesId, 50)}`);
  $("#delete-series-info-type").text(`Type: ${typeText}`);
  $("#delete-series-info-patient").text(`Patient: ${patientName || "N/A"}`);
  showModal("#confirmationModal");
}

// Delete result
function deleteResult(seriesId, calculationType) {
  setStatus("Deleting...", "loading");
  const url = calculationType
    ? `/cho-results/${seriesId}?calculation_type=${calculationType}`
    : `/cho-results/${seriesId}`;

  $.ajax({
    url: url,
    type: "DELETE",
    success: function (data) {
      setStatus("Successfully deleted", "success");
      loadData();
      hideModal("#confirmationModal");
      alert("Result deleted successfully");
    },
    error: function (xhr, status, error) {
      const errorMsg = xhr.responseJSON?.message || error;
      setStatus(`Error deleting: ${errorMsg}`, "error");
      hideModal("#confirmationModal");
      alert("Error deleting result");
    },
  });
}

// Export to CSV
function exportToCSV() {
  setStatus("Exporting to CSV...", "loading");
  window.open("/cho-results-export", "_blank");
  setStatus("Export initiated", "success");
}

// Helper functions for modal animations
function showModal(modalSelector) {
  const modal = $(modalSelector);
  modal.show().addClass("show");
}

function hideModal(modalSelector) {
  const modal = $(modalSelector);
  modal.removeClass("show");
  setTimeout(() => {
    modal.hide();
  }, 300);
  // Remove the event listener for exporting individual results
  $("#export-individual-result").off("click");
}

// Helper functions
function formatNumber(value, decimals = 2) {
  if (value === null || value === undefined) return "N/A";
  return Number(value).toFixed(decimals);
}

function formatDateTime(dateStr) {
  if (!dateStr) return "N/A";
  return new Date(dateStr).toLocaleString();
}

function truncateText(text, maxLength) {
  if (!text) return "N/A";
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength) + "...";
}

function setStatus(message, type) {
  const statusEl = $("#status");
  statusEl.text(message);
  statusEl.removeClass("error success loading");
  if (type) statusEl.addClass(type);
}
