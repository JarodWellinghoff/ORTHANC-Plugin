// Stable scrollable viewports on every dashboard route. Keep server pagination
// for large catalogs, but remove row-size controls rather than stretching pages.
export const withScrollableTables = (components) => ({
  ...components,
  MuiDataGrid: {
    ...components.MuiDataGrid,
    defaultProps: { ...components.MuiDataGrid?.defaultProps, autoHeight: false },
    styleOverrides: { ...components.MuiDataGrid?.styleOverrides,
      root: (ownerState) => ({
        ...(typeof components.MuiDataGrid?.styleOverrides?.root === "function"
          ? components.MuiDataGrid.styleOverrides.root(ownerState) : components.MuiDataGrid?.styleOverrides?.root),
        height: 560, minHeight: 320, maxHeight: 560, flexShrink: 0,
        "& .MuiTablePagination-selectLabel, & .MuiTablePagination-input": { display: "none" },
      }),
    },
  },
  MuiTableContainer: { ...components.MuiTableContainer,
    styleOverrides: { ...components.MuiTableContainer?.styleOverrides, root: { maxHeight: 560, overflow: "auto" } } },
  MuiTable: { ...components.MuiTable, defaultProps: { ...components.MuiTable?.defaultProps, stickyHeader: true } },
});
