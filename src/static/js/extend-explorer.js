// Updated extend-explorer.js - Single CHO Analysis Button with Modal

// Add CHO button to the lookup page
$("#lookup").live("pagebeforecreate", function () {
  $("head").append(`
    <script src="/static/js/lookup.js"></script>
  `);
});

$("#series").live("pagebeforecreate", function () {
  $("head").append(`
    <script src="/static/js/series.js"></script>
  `);
});

$("#study").live("pagebeforecreate", function () {
  $("head").append(`
    <script src="/static/js/study.js"></script>
  `);
});
