// import Box from "@mui/material/Box";
// import { NumberField } from "@base-ui-components/react/number-field";
// import Stack from "@mui/material/Stack";
// import Typography from "@mui/material/Typography";
// import IconButton from "@mui/material/IconButton";
// import KeyboardArrowDownRoundedIcon from "@mui/icons-material/KeyboardArrowDownRounded";
// import KeyboardArrowUpRoundedIcon from "@mui/icons-material/KeyboardArrowUpRounded";
// import { FormControl } from "@mui/material";

// export default function ChoNumberField(
//   {
//     id,
//     label,
//     helperText,
//     errorText,
//     error: errorProp = false,
//     required = false,
//     disabled = false,
//     value,
//     defaultValue,
//     min,
//     max,
//     step = 0.01,
//     placeholder,
//     readOnly = false,
//     onChange,
//     onCommit,
//     fullWidth = true,
//     size = "medium",
//     sx,
//     ...numberFieldProps
//   },
//   ref
// ) {
//   const HEIGHT = size === "small" ? 20 : 40; // MUI Select heights
//   const END_COL_W = size === "small" ? 32 : 36; // like the chevron area
//   const PX = size === "small" ? 1.25 : 1.5; // horizontal padding (theme spacing units)

//   const error = errorProp || Boolean(errorText);
//   const stackSx = [
//     { width: fullWidth ? "100%" : "auto", height: HEIGHT },
//     ...(Array.isArray(sx) ? sx : sx ? [sx] : []),
//   ];
//   return (
//     <Stack
//       spacing={-1.5}
//       sx={stackSx}
//       component={FormControl}
//       //   className='MuiFormControl-root MuiFormControl-fullWidth css-ytlejw-MuiFormControl-root'
//     >
//       <NumberField.Root
//         ref={ref}
//         className='MuiInputBase-root MuiOutlinedInput-root MuiInputBase-colorPrimary MuiInputBase-fullWidth MuiInputBase-formControl MuiSelect-root css-xk652r-MuiInputBase-root-MuiOutlinedInput-root-MuiSelect-root'
//         value={value}
//         defaultValue={defaultValue}
//         min={min}
//         max={max}
//         step={step}
//         disabled={disabled}
//         required={required}
//         readOnly={readOnly}
//         onValueChange={(nextValue, details) => {
//           onChange?.(nextValue, details);
//         }}
//         onValueCommitted={(nextValue, details) => {
//           onCommit?.(nextValue, details);
//         }}
//         {...numberFieldProps}>
//         {label ? (
//           <Typography
//             variant='caption'
//             color='text.secondary'
//             component='label'
//             className='MuiFormLabel-root MuiInputLabel-root MuiInputLabel-formControl MuiInputLabel-animated MuiInputLabel-shrink MuiInputLabel-outlined MuiFormLabel-colorPrimary MuiFormLabel-filled MuiInputLabel-root MuiInputLabel-formControl MuiInputLabel-animated MuiInputLabel-shrink MuiInputLabel-outlined css-1peeo22-MuiFormLabel-root-MuiInputLabel-root'
//             for={id}
//             //   sx={{
//             //     fontWeight: 400,
//             //     fontSize: "0.875rem",
//             //     lineHeight: 1.5,
//             //     bgcolor: (t) => (t.vars || t).palette.background.paper,
//             //     width: "fit-content",
//             //     "&&": {
//             //       marginLeft: 1.5,
//             //       zIndex: 1,
//             //       px: 1,
//             //     },
//             //   }}
//           >
//             {label}
//             {required ? " *" : ""}
//           </Typography>
//         ) : null}
//         <NumberField.Group
//           render={(groupProps, state) => {
//             const { children, ...rest } = groupProps;
//             return (
//               <Box
//                 {...rest}
//                 sx={{
//                   display: "grid",
//                   gridTemplateAreas: `"input end"`,
//                   gridTemplateColumns: `1fr ${END_COL_W}px`,
//                   borderRadius: 1, // match Select
//                   border: "1px solid",
//                   borderColor: (t) =>
//                     error
//                       ? t.palette.error.main
//                       : state.focused
//                       ? t.palette.primary.main
//                       : t.palette.divider,
//                   //   bgcolor: (t) =>
//                   //     disabled
//                   //       ? t.palette.action.disabledBackground
//                   //       : t.palette.background.paper,
//                   transition: (t) =>
//                     t.transitions.create(["border-color", "box-shadow"], {
//                       duration: t.transitions.duration.shorter,
//                     }),
//                   boxShadow: (t) =>
//                     state.focused && !error
//                       ? `0 0 0 3px ${t.palette.primary.main}1F`
//                       : "none",
//                   "&:hover":
//                     !disabled && !error
//                       ? { borderColor: "text.primary" }
//                       : undefined,
//                   //   overflow: "hidden",
//                   //   height: HEIGHT,
//                 }}>
//                 {children}
//               </Box>
//             );
//           }}>
//           <NumberField.Input
//             render={(inputProps) => (
//               <Box
//                 {...inputProps}
//                 component='input'
//                 placeholder={placeholder}
//                 sx={{
//                   gridArea: "input",
//                   minWidth: 0,
//                   border: 0,
//                   outline: 0,
//                   bgcolor: "transparent",
//                   color: disabled ? "text.disabled" : "text.primary",
//                   fontFamily: (t) => t.typography.fontFamily,
//                   fontSize: size === "small" ? 14 : 16,
//                   lineHeight: 1.4375, // MUI input line-height
//                   px: PX,
//                   height: "100%",
//                   "&::placeholder": { color: "text.disabled", opacity: 1 },
//                   "&::-webkit-outer-spin-button, &::-webkit-inner-spin-button":
//                     {
//                       WebkitAppearance: "none",
//                       margin: 0,
//                     },
//                   "&[type=number]": { MozAppearance: "textfield" },
//                   //   borderRight: "1px solid",
//                   borderColor: error ? "error.main" : "divider",
//                 }}
//               />
//             )}
//           />

//           {/* End column mimics Select’s chevron box */}
//           <Box
//             sx={{
//               gridArea: "end",
//               display: "grid",
//               gridTemplateRows: "1fr 1fr",
//               alignItems: "stretch",
//               justifyItems: "stretch",
//               bgcolor: "transparent",
//             }}>
//             <NumberField.Increment
//               aria-label={`Increase ${label ?? "value"}`}
//               render={(incrementProps) => (
//                 <IconButton
//                   {...incrementProps}
//                   size='small'
//                   disableRipple
//                   sx={{
//                     borderRadius: 0,
//                     border: 0,
//                     height: "100%",
//                     color: disabled ? "text.disabled" : "text.secondary",
//                     // borderBottom: "1px solid",
//                     borderColor: "transparent",
//                     "&:hover": {
//                       bgcolor: (t) =>
//                         disabled ? "transparent" : t.palette.action.hover,
//                     },
//                   }}>
//                   <KeyboardArrowUpRoundedIcon fontSize='inherit' />
//                 </IconButton>
//               )}
//             />
//             <NumberField.Decrement
//               aria-label={`Decrease ${label ?? "value"}`}
//               render={(decrementProps) => (
//                 <IconButton
//                   {...decrementProps}
//                   size='small'
//                   disableRipple
//                   sx={{
//                     borderRadius: 0,
//                     border: 0,
//                     height: "100%",
//                     color: disabled ? "text.disabled" : "text.secondary",
//                     "&:hover": {
//                       bgcolor: (t) =>
//                         disabled ? "transparent" : t.palette.action.hover,
//                     },
//                   }}>
//                   <KeyboardArrowDownRoundedIcon fontSize='inherit' />
//                 </IconButton>
//               )}
//             />
//           </Box>
//         </NumberField.Group>
//       </NumberField.Root>
//       {helperText ? (
//         <Typography variant='caption' color='text.secondary'>
//           {helperText}
//         </Typography>
//       ) : null}
//       {errorText ? (
//         <Typography variant='caption' color='error.main'>
//           {errorText}
//         </Typography>
//       ) : null}
//     </Stack>
//   );
// }
