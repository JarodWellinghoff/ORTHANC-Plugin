$("#lookup").live("pagebeforeshow", function () {
  $("#open-dash").remove();

  var dashboardButton = $("<fieldset>")
    .attr("id", "open-dash")
    .addClass("ui-grid-b")
    .append($("<div>").addClass("ui-block-a"))
    .append(
      $("<div>")
        .addClass("ui-block-b")
        .append(
          $("<a>")
            .attr("data-role", "button")
            .attr("href", "#")
            .attr("data-icon", "forward")
            .attr("data-theme", "a")
            .text("CHO Analysis Dashboard")
            .button()
            .click(function (e) {
              window.open("/cho-dashboard", "_blank");
            })
        )
    );

  dashboardButton.insertAfter($("#lookup-result"));
});
