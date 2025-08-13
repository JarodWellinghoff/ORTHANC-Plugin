import numpy as np
from typing import Generator, Tuple, Dict, Any, List, Union
from skimage import feature

class CTSlidingWindow:
    def __init__(self, 
                 window_length: Union[float, int] = 15.0,
                 step_size: Union[float, int] = 5.0,
                 unit: str = 'cm',  # 'cm' or 'slices' (for backward compatibility)
                 window_unit: str | None = None,  # 'cm' or 'slices' (overrides unit for window_length)
                 step_unit: str | None = None,    # 'cm' or 'slices' (overrides unit for step_size)
                 sigma: float = 1.0,
                 padding_size: int = 0,
                 use_cache: bool = True
                 ): 
        """
        Initialize CT Sliding Window with support for mixed units.
        
        Parameters:
        - window_length: Window size in cm (float) or number of slices (int)
        - step_size: Step size in cm (float) or number of slices (int)
        - unit: 'cm' or 'slices' (applied to both if window_unit/step_unit not specified)
        - window_unit: 'cm' or 'slices' (unit for window_length, overrides unit)
        - step_unit: 'cm' or 'slices' (unit for step_size, overrides unit)
        - sigma: Sigma value for Canny edge detection
        - padding_size: Padding size for integral image calculations
        
        Examples:
        - CTSlidingWindow(15.0, 5.0, unit='cm')  # Both in cm
        - CTSlidingWindow(10, 3, unit='slices')  # Both in slices  
        - CTSlidingWindow(1, 0.3, window_unit='slices', step_unit='cm')  # Mixed units
        """
        self.window_length = window_length
        self.step_size = step_size
        self.sigma = sigma
        self.padding_size = padding_size
        self.use_cache = use_cache

        # Determine units for each parameter
        self.window_unit = (window_unit or unit).lower()
        self.step_unit = (step_unit or unit).lower()
        
        # Validate units
        valid_units = ['cm', 'slices']
        if self.window_unit not in valid_units:
            raise ValueError("window_unit must be either 'cm' or 'slices'")
        if self.step_unit not in valid_units:
            raise ValueError("step_unit must be either 'cm' or 'slices'")
        
        # Validate parameter types and values
        if self.window_unit == 'slices':
            if not isinstance(window_length, int) or window_length <= 0:
                raise ValueError("window_length must be a positive integer when window_unit='slices'")
        else:  # cm
            if window_length <= 0:
                raise ValueError("window_length must be positive when window_unit='cm'")
        
        if self.step_unit == 'slices':
            if not isinstance(step_size, int) or step_size <= 0:
                raise ValueError("step_size must be a positive integer when step_unit='slices'")
        else:  # cm
            if step_size <= 0:
                raise ValueError("step_size must be positive when step_unit='cm'")
        
        # Store original unit parameter for backward compatibility
        self.unit = unit.lower()
        
        # Initialize state variables
        self.num_windows = None
        self.positions = None
        self.min_pos = None
        self.max_pos = None
        self.total_length = None
        self.total_slices = None
        self.slice_spacings = None  # For mixed unit calculations
        
        # Cache for computed slices
        self.pixel_array_cache = {}
        self.integral_edges_cache = {}
        self.integral_images_cache = {}
        self.integral_images_square_cache = {}

    def _get_slice_positions(self, dicom_slices: list) -> np.ndarray:
        """Extract z-positions from DICOM headers."""
        positions = []
        for dcm in dicom_slices:
            # SliceLocation gives z-coordinate in mm
            z_pos = float(dcm.SliceLocation)
            positions.append(z_pos / 10.0)  # Convert mm to cm
        return np.array(positions)
    
    def _find_slices_in_range(self, positions: np.ndarray, 
                             start_cm: float, end_cm: float) -> List[int]:
        """Find slice indices within the given cm range."""
        # Include slices whose positions fall within [start, end)
        mask = (positions >= start_cm) & (positions < end_cm)
        return np.where(mask)[0].tolist()
    
    def _find_slices_by_index_range(self, start_idx: int, end_idx: int, 
                                   total_slices: int) -> List[int]:
        """Find slice indices within the given index range."""
        start_idx = max(0, start_idx)
        end_idx = min(total_slices, end_idx)
        return list(range(start_idx, end_idx))
    
    def _integral_image(self, image, window_size=[3, 3], padding='constant'):
        """
        AVERAGEFILTER 2-D mean filtering.  
        Performs mean filtering of a 2-dimensional matrix/image using the integral image method.    

        Parameters:
        - image: 2D array-like, the input image to be filtered.
        - window_size: tuple, (M, N) defines the vertical and horizontal window size.
        - padding: str, can be 'constant', 'reflect', 'symmetric', or 'wrap'.   

        Returns:
        - image: 2D array, the filtered image.
        """
        
        if len(image.shape) != 2:
            raise ValueError("The input image must be a two-dimensional array.")
        
        # Set up the window size
        m, n = window_size    

        # Pad the image
        pad_width = ((m // 2, m // 2), (n // 2, n // 2))  # Calculate padding around the image

        if padding == 'circular':
            padded_image = np.pad(image, pad_width, mode='wrap')

        elif padding == 'replicate':
            padded_image = np.pad(image, pad_width, mode='edge')

        elif padding == 'symmetric':
            padded_image = np.pad(image, pad_width, mode='symmetric')

        else:  # Default is 'constant' padding (zero padding)
            padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)

        # Convert the image to float
        imageD = padded_image.astype(np.float64)
        

        # Calculate the integral image
        t = np.cumsum(np.cumsum(imageD, axis=0), axis=1)

        # Cast the resulting image back to the original type
        return t
    
    def _compute_slice(self, slice_idx: int, dicom_slices: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the pixel array for a specific slice index.

        Args:
            slice_idx (int): Index of the slice to compute.
            dicom_slices (List): List of DICOM slices.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the pixel array, integral images, and edges.
        """
        if self.use_cache and slice_idx in self.pixel_array_cache and slice_idx in self.integral_edges_cache and slice_idx in self.integral_images_cache and slice_idx in self.integral_images_square_cache:
        # if self.use_cache and slice_idx in self.pixel_array_cache:
            print(f'Using cached data for slice {slice_idx}')
            edges = feature.canny(self.pixel_array_cache[slice_idx].astype(float), sigma=self.sigma)
            return self.pixel_array_cache[slice_idx], self.integral_edges_cache[slice_idx], self.integral_images_cache[slice_idx], self.integral_images_square_cache[slice_idx], edges
            # return self.pixel_array_cache[slice_idx]

        # Convert DICOM to pixel array
        # print(f'Calculating data for slice {slice_idx}')
        pixel_array = dicom_slices[slice_idx].pixel_array
        
        # Apply any rescaling if needed
        if hasattr(dicom_slices[slice_idx], 'RescaleSlope') and hasattr(dicom_slices[slice_idx], 'RescaleIntercept'):
            slope = float(dicom_slices[slice_idx].RescaleSlope)
            intercept = float(dicom_slices[slice_idx].RescaleIntercept)
            pixel_array = pixel_array * slope + intercept


        # Apply Canny edge detection
        # Note: Canny edge detection requires float input, so we convert pixel_array to float
        edges = feature.canny(pixel_array.astype(float), sigma=self.sigma)
        integral_edges = self._integral_image(edges, [self.padding_size, self.padding_size], 'replicate')
        integral_images = self._integral_image(pixel_array, [self.padding_size, self.padding_size], 'replicate')
        integral_images_square = self._integral_image(pixel_array**2, [self.padding_size, self.padding_size], 'replicate')

        pixel_array = pixel_array.astype(np.int32)
        integral_edges = integral_edges.astype(np.int32)
        integral_images = integral_images.astype(np.int32)
        integral_images_square = integral_images_square.astype(np.float64)

        # Cache the result
        if self.use_cache:
            self.pixel_array_cache[slice_idx] = pixel_array
            self.integral_edges_cache[slice_idx] = integral_edges
            self.integral_images_cache[slice_idx] = integral_images
            self.integral_images_square_cache[slice_idx] = integral_images_square

        return pixel_array, integral_edges, integral_images, integral_images_square, edges
        # return pixel_array

    def get_num_windows(self, dicom_slices: list) -> int:
        """Get the number of sliding windows."""
        if self.num_windows is not None:
            return self.num_windows

        self.total_slices = len(dicom_slices)
        self.positions = self._get_slice_positions(dicom_slices)
        self.min_pos = self.positions.min()
        self.max_pos = self.positions.max()
        self.total_length = self.max_pos - self.min_pos
        
        # Calculate slice spacings for mixed unit conversions
        self.slice_spacings = np.diff(self.positions)  # cm between consecutive slices
        
        # Handle different unit combinations
        if self.window_unit == 'cm' and self.step_unit == 'cm':
            # Both in cm - original logic
            self.num_windows = int((self.total_length - self.window_length) / self.step_size) + 1
            
        elif self.window_unit == 'slices' and self.step_unit == 'slices':
            # Both in slices - original logic
            if self.window_length > self.total_slices:
                self.num_windows = 0
            else:
                self.num_windows = int((self.total_slices - self.window_length) / self.step_size) + 1
                
        elif self.window_unit == 'slices' and self.step_unit == 'cm':
            # Window in slices, step in cm
            # Need to simulate stepping through in cm to count windows
            self.num_windows = self._count_mixed_windows_slice_window_cm_step(dicom_slices)
            
        elif self.window_unit == 'cm' and self.step_unit == 'slices':
            # Window in cm, step in slices
            # Need to simulate stepping through in slices to count windows
            self.num_windows = self._count_mixed_windows_cm_window_slice_step(dicom_slices)
        
        if self.num_windows is None:
            self.num_windows = 0
        else:
            self.num_windows = max(0, self.num_windows)  # Ensure non-negative
        return self.num_windows
    
    def _count_mixed_windows_slice_window_cm_step(self, dicom_slices: list) -> int:
        """Count windows when window_length is in slices and step_size is in cm."""
        if self.window_length > self.total_slices:
            return 0
            
        count = 0
        current_pos = self.min_pos
        
        while True:
            # Find the end position for this window (window_length slices from current position)
            start_slice_idx = self._find_nearest_slice_index(current_pos)
            
            # Check if we have enough slices from this position
            if start_slice_idx + self.window_length > self.total_slices:
                break
                
            count += 1
            current_pos += self.step_size  # Step in cm
            
            # Safety check to prevent infinite loops
            if current_pos > self.max_pos:
                break
                
        return count
    
    def _count_mixed_windows_cm_window_slice_step(self, dicom_slices: list) -> int:
        """Count windows when window_length is in cm and step_size is in slices."""
        count = 0
        current_slice_idx = 0
        
        while current_slice_idx < self.total_slices:
            # Check if we can fit a window of window_length cm from this slice
            start_pos = self.positions[current_slice_idx]
            end_pos = start_pos + self.window_length
            
            # Check if the window extends beyond the scan
            if end_pos > self.max_pos:
                break
                
            # Find slices within this cm window
            slice_indices = self._find_slices_in_range(self.positions, start_pos, end_pos)
            if len(slice_indices) > 0:
                count += 1
                
            # Step by step_size slices
            current_slice_idx += self.step_size
            
        return count
    
    def _find_nearest_slice_index(self, position_cm: float) -> int:
        """Find the slice index nearest to the given position in cm."""
        distances = np.abs(self.positions - position_cm)
        return int(np.argmin(distances))

    def generate_windows(self, dicom_slices: list) -> Generator[Tuple[Dict[str, Any], Dict[str, Any]], None, None]:
        """Generate sliding windows over CT scan using various unit combinations."""
        # Get actual positions from DICOM headers and calculate num_windows
        if self.num_windows is None or self.positions is None or self.min_pos is None or self.max_pos is None:
            self.get_num_windows(dicom_slices)

        # Dispatch to appropriate method based on unit combination
        if self.window_unit == 'cm' and self.step_unit == 'cm':
            yield from self._generate_windows_cm_cm(dicom_slices)
        elif self.window_unit == 'slices' and self.step_unit == 'slices':
            yield from self._generate_windows_slices_slices(dicom_slices)
        elif self.window_unit == 'slices' and self.step_unit == 'cm':
            yield from self._generate_windows_slices_cm(dicom_slices)
        elif self.window_unit == 'cm' and self.step_unit == 'slices':
            yield from self._generate_windows_cm_slices(dicom_slices)
    
    def _generate_windows_cm_cm(self, dicom_slices: list) -> Generator[Tuple[Dict[str, Any], Dict[str, Any]], None, None]:
        """Generate windows when both window_length and step_size are in cm."""
        for window_idx in range(self.num_windows):
            start_cm = self.min_pos + (window_idx * self.step_size)
            end_cm = start_cm + self.window_length
            
            # Skip if window extends beyond scan
            if end_cm > self.max_pos:
                break
            
            # Find slices within this window
            slice_indices = self._find_slices_in_range(self.positions, start_cm, end_cm)

            if not slice_indices:
                continue  # Skip empty windows
            
            # Build window by combining slices
            window_array, slice_positions = self._build_window(slice_indices, dicom_slices)
            
            # Window metadata
            metadata = self._create_metadata(window_idx, slice_indices, slice_positions, start_cm, end_cm, start_cm, end_cm)
            yield window_array, metadata
    
    def _generate_windows_slices_slices(self, dicom_slices: list) -> Generator[Tuple[Dict[str, Any], Dict[str, Any]], None, None]:
        """Generate windows when both window_length and step_size are in slices."""
        for window_idx in range(self.num_windows):
            start_idx = window_idx * self.step_size
            end_idx = start_idx + self.window_length
            
            # Skip if window extends beyond available slices
            if end_idx > self.total_slices:
                break
            
            # Get slice indices for this window
            slice_indices = self._find_slices_by_index_range(start_idx, end_idx, self.total_slices)

            if not slice_indices:
                continue  # Skip empty windows
            
            # Build window by combining slices
            window_array, slice_positions = self._build_window(slice_indices, dicom_slices)
            
            # Calculate cm-based positions for metadata
            start_cm = slice_positions[0] if slice_positions else 0
            end_cm = slice_positions[-1] if slice_positions else 0
            
            # Window metadata
            metadata = self._create_metadata(window_idx, slice_indices, slice_positions, 
                                           start_cm, end_cm, start_idx, end_idx)
            yield window_array, metadata
    
    def _generate_windows_slices_cm(self, dicom_slices: list) -> Generator[Tuple[Dict[str, Any], Dict[str, Any]], None, None]:
        """Generate windows when window_length is in slices and step_size is in cm."""
        window_idx = 0
        current_pos = self.min_pos
        
        while window_idx < self.num_windows:
            # Find the starting slice index for this position
            start_slice_idx = self._find_nearest_slice_index(current_pos)
            
            # Check if we have enough slices from this position
            if start_slice_idx + self.window_length > self.total_slices:
                break
            
            # Get slice indices for this window (window_length slices)
            slice_indices = list(range(start_slice_idx, start_slice_idx + self.window_length))
            
            # Build window by combining slices
            window_array, slice_positions = self._build_window(slice_indices, dicom_slices)
            
            # Calculate cm positions
            start_cm = slice_positions[0] if slice_positions else current_pos
            end_cm = slice_positions[-1] if slice_positions else current_pos
            
            # Window metadata
            metadata = self._create_metadata(window_idx, slice_indices, slice_positions, 
                                           start_cm, end_cm, current_pos, current_pos + self.step_size)
            yield window_array, metadata
            
            # Step by step_size cm
            current_pos += self.step_size
            window_idx += 1
            
            # Safety check
            if current_pos > self.max_pos:
                break
    
    def _generate_windows_cm_slices(self, dicom_slices: list) -> Generator[Tuple[Dict[str, Any], Dict[str, Any]], None, None]:
        """Generate windows when window_length is in cm and step_size is in slices."""
        window_idx = 0
        current_slice_idx = 0

        if self.num_windows is None or self.positions is None or self.min_pos is None or self.max_pos is None:
            self.get_num_windows(dicom_slices)
        
        if self.num_windows is None or self.total_slices is None:
            return
        
        # Ensure we have valid positions    
        if self.positions is None or len(self.positions) == 0:
            return
        
        while window_idx < self.num_windows and current_slice_idx < self.total_slices:
            # Calculate cm window from current slice position
            start_pos = self.positions[current_slice_idx]
            end_pos = start_pos + self.window_length
            
            # Check if the window extends beyond the scan
            if end_pos > self.max_pos:
                break
            
            # Find slices within this cm window
            slice_indices = self._find_slices_in_range(self.positions, start_pos, end_pos)
            
            if not slice_indices:
                current_slice_idx += self.step_size
                continue
            
            # Build window by combining slices
            window_array, slice_positions = self._build_window(slice_indices, dicom_slices)
            
            # Window metadata
            metadata = self._create_metadata(window_idx, slice_indices, slice_positions, 
                                           start_pos, end_pos, current_slice_idx, current_slice_idx + self.step_size)
            yield window_array, metadata
            
            # Step by step_size slices
            current_slice_idx += self.step_size
            window_idx += 1
    
    def _create_metadata(self, window_idx: int, slice_indices: List[int], slice_positions: List[float],
                        start_cm: float, end_cm: float, logical_start, logical_end) -> Dict[str, Any]:
        """Create comprehensive metadata for a window."""
        
        # Calculate step size in cm for this specific window (for mixed units)
        if window_idx > 0 and hasattr(self, '_previous_start_cm'):
            actual_step_size_cm = start_cm - self._previous_start_cm
        else:
            actual_step_size_cm = None
        self._previous_start_cm = start_cm
        
        # Calculate step size in slices for this specific window (for mixed units)
        if window_idx > 0 and hasattr(self, '_previous_start_idx'):
            actual_step_size_slices = slice_indices[0] - self._previous_start_idx
        else:
            actual_step_size_slices = None
        self._previous_start_idx = slice_indices[0] if slice_indices else 0
        
        metadata = {
            'window_idx': window_idx,
            'num_windows': self.num_windows,
            
            # Unit information
            'window_unit': self.window_unit,
            'step_unit': self.step_unit,
            'unit': self.unit,  # For backward compatibility
            
            # Window configuration
            'window_length': self.window_length,
            'step_size': self.step_size,
            'window_length_config_unit': self.window_unit,
            'step_size_config_unit': self.step_unit,
            
            # Actual measurements for this window
            'window_length_slices': len(slice_indices),
            'window_length_cm': abs(end_cm - start_cm) if len(slice_positions) > 1 else 0,
            'step_size_slices': actual_step_size_slices,
            'step_size_cm': actual_step_size_cm,
            
            # Position information
            'actual_start_cm': start_cm,
            'relative_start_cm': start_cm - self.min_pos if self.min_pos is not None else start_cm,
            'start_idx': slice_indices[0] if slice_indices else 0,
            'end_idx': slice_indices[-1] if slice_indices else 0,
            'actual_end_cm': end_cm,
            'relative_end_cm': end_cm - self.min_pos if self.min_pos is not None else end_cm,
            'actual_mid_cm': (start_cm + end_cm) / 2,
            'relative_mid_cm': (start_cm + end_cm) / 2 - self.min_pos if self.min_pos is not None else (start_cm + end_cm) / 2,
            
            # Logical positions (what was requested)
            'logical_start': logical_start,
            'logical_end': logical_end,
            
            # Scan information
            'total_length_cm': self.total_length,
            'total_slices': self.total_slices,
            
            # Slice information
            'slice_indices': slice_indices,
            'slice_positions_cm': slice_positions,
            'num_slices': len(slice_indices),
            'actual_coverage_cm': abs(slice_positions[-1] - slice_positions[0]) if len(slice_positions) > 1 else 0,
            
            # Processing parameters
            'sigma': self.sigma,
            'padding_size': self.padding_size
        }
        
        return metadata
    
    def _build_window(self, slice_indices: List[int], dicom_slices: list) -> Tuple[Dict[str, Any], List[float]]:
        """Build window data from slice indices."""
        pixel_arrays = []
        integral_edges_arrays = []
        integral_images_arrays = []
        integral_images_square_arrays = []
        # edges = None
        slice_positions = []
        
        for idx in slice_indices:
            pixel_array, integral_edges, integral_images, integral_images_square, edges = self._compute_slice(idx, dicom_slices)
            # pixel_array = self._compute_slice(idx, dicom_slices)
            pixel_arrays.append(pixel_array)
            integral_edges_arrays.append(integral_edges)
            integral_images_arrays.append(integral_images)
            integral_images_square_arrays.append(integral_images_square)
            slice_positions.append(self.positions[idx])
        
        # Stack slices into 3D arrays
        window_pixel_array = np.stack(pixel_arrays, axis=-1)
        window_integral_edges = np.stack(integral_edges_arrays, axis=-1)
        window_integral_images = np.stack(integral_images_arrays, axis=-1)
        window_integral_images_square = np.stack(integral_images_square_arrays, axis=-1)

        window_array = {
            'pixel_array': window_pixel_array,
            'integral_edges': window_integral_edges,
            'integral_images': window_integral_images,
            'integral_images_square': window_integral_images_square,
            'edges': edges
        }
        
        return window_array, slice_positions
            
    def clear_cache(self, keep_indices: List[int] | None = None):
        """Clear cache, optionally keeping specific slice indices."""
        if not self.use_cache:
            return

        if keep_indices is None:
            self.pixel_array_cache.clear()
            self.integral_edges_cache.clear()
            self.integral_images_cache.clear()
            self.integral_images_square_cache.clear()
        else:
            # Keep only specified indices
            keys_to_remove = [k for k in self.pixel_array_cache.keys() if k not in keep_indices]
            for k in keys_to_remove:
                del self.pixel_array_cache[k]

            keys_to_remove = [k for k in self.integral_edges_cache.keys() if k not in keep_indices]
            for k in keys_to_remove:
                del self.integral_edges_cache[k]

            keys_to_remove = [k for k in self.integral_images_cache.keys() if k not in keep_indices]
            for k in keys_to_remove:
                del self.integral_images_cache[k]

            keys_to_remove = [k for k in self.integral_images_square_cache.keys() if k not in keep_indices]
            for k in keys_to_remove:
                del self.integral_images_square_cache[k]


# Example usage:
def example_usage():
    """Example usage showing cm, slice-based, and mixed unit approaches."""
    
    # Example 1: CM-based windows (original approach)
    print("=== CM-based windows ===")
    window_gen_cm = CTSlidingWindow(
        window_length=15.0,  # 15 cm
        step_size=5.0,       # 5 cm step
        unit='cm'
    )
    
    # Example 2: Slice-based windows
    print("=== Slice-based windows ===")
    window_gen_slices = CTSlidingWindow(
        window_length=10,    # 10 slices
        step_size=3,         # 3 slice step
        unit='slices'
    )
    
    # Example 3: Mixed units - slice window with cm step (your use case)
    print("=== Mixed: 1 slice window, 3mm step ===")
    window_gen_mixed1 = CTSlidingWindow(
        window_length=1,     # 1 slice
        step_size=0.3,       # 3mm = 0.3cm step
        window_unit='slices',
        step_unit='cm'
    )
    
    # Example 4: Mixed units - cm window with slice step
    print("=== Mixed: 2cm window, 2 slice step ===")
    window_gen_mixed2 = CTSlidingWindow(
        window_length=2.0,   # 2 cm
        step_size=2,         # 2 slice step
        window_unit='cm',
        step_unit='slices'
    )
    
    # Example 5: Backward compatibility - using old 'unit' parameter
    print("=== Backward compatible ===")
    window_gen_compat = CTSlidingWindow(
        window_length=15.0,
        step_size=5.0,
        unit='cm'  # Both window and step will use cm
    )
    
    # Usage example (assuming dicom_slices is available):
    """
    # For mixed units example (1 slice every 3mm)
    for window, metadata in window_gen_mixed1.generate_windows(dicom_slices):
        print(f"Mixed Window {metadata['window_idx']}: "
              f"{metadata['window_length_slices']} slice(s) at {metadata['actual_start_cm']:.1f}cm")
        print(f"  Step: {metadata['step_size_cm']:.1f}cm from previous window")
        print(f"  Slice indices: {metadata['slice_indices']}")
        print(f"  Actual coverage: {metadata['actual_coverage_cm']:.1f}cm")
        print()
    
    # For cm window with slice step
    for window, metadata in window_gen_mixed2.generate_windows(dicom_slices):
        print(f"Mixed Window {metadata['window_idx']}: "
              f"{metadata['window_length_cm']:.1f}cm window starting at slice {metadata['start_idx']}")
        print(f"  Step: {metadata['step_size_slices']} slices from previous window")
        print(f"  CM range: {metadata['actual_start_cm']:.1f}cm - {metadata['actual_end_cm']:.1f}cm")
        print(f"  Includes {metadata['num_slices']} slices")
        print()
    """

def create_common_configurations():
    """Factory functions for common mixed-unit configurations."""
    
    def single_slice_sampling(step_mm: float):
        """Create config for 1 slice every X mm sampling."""
        return CTSlidingWindow(
            window_length=1,
            step_size=step_mm / 10.0,  # Convert mm to cm
            window_unit='slices',
            step_unit='cm'
        )
    
    def thin_slice_cm_windows(window_cm: float, step_slices: int):
        """Create config for cm windows with slice stepping (good for thin slice data)."""
        return CTSlidingWindow(
            window_length=window_cm,
            step_size=step_slices,
            window_unit='cm',
            step_unit='slices'
        )
    
    def anatomical_sampling(anatomy_length_cm: float, sampling_interval_mm: float):
        """Create config for consistent anatomical sampling."""
        return CTSlidingWindow(
            window_length=anatomy_length_cm,
            step_size=sampling_interval_mm / 10.0,  # Convert mm to cm
            unit='cm'  # Both in cm for anatomical consistency
        )
    
    return {
        'single_slice_3mm': single_slice_sampling(3.0),      # 1 slice every 3mm
        'single_slice_5mm': single_slice_sampling(5.0),      # 1 slice every 5mm
        'thin_slice_2cm': thin_slice_cm_windows(2.0, 1),     # 2cm windows, 1 slice step
        'anatomical_1cm': anatomical_sampling(1.0, 2.5),     # 1cm windows, 2.5mm step
    }

if __name__ == "__main__":
    example_usage()
    
    # Show common configurations
    configs = create_common_configurations()
    print("\n=== Common Configurations ===")
    for name, config in configs.items():
        print(f"{name}: window={config.window_length}{config.window_unit}, "
              f"step={config.step_size}{config.step_unit}")
