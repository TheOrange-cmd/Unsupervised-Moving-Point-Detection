import sys
import os
import numpy as np

# --- Add the build directory to Python's path ---
# Adjust this path based on your build directory structure
# Example: if build is parallel to src, and module is in build/
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './build'))
# Or if module is in build/python/
# module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../build/python'))
print(f"Adding to sys.path: {module_path}")
sys.path.insert(0, module_path)

try:
    import mpy_detector as mdet
    print("Successfully imported mpy_detector")
except ImportError as e:
    print(f"Error importing mpy_detector: {e}")
    print("Check if the module (.so/.pyd) is in the build directory and the path is correct.")
    sys.exit(1)

# --- Test basic functionality ---

# 1. Test config loading
config_file = '/home/drugge/Unsupervised-Moving-Point-Detection/test/config/test_full_config.yaml' # Adjust path as needed
print(f"\nLoading config from: {config_file}")
try:
    params = mdet.load_config(config_file)
    print("Config loaded successfully.")
    print(f"Type of loaded object: {type(params)}")

    # Check if it's the correct type
    assert isinstance(params, mdet.DynObjFilterParams)
    print("Loaded object is an instance of DynObjFilterParams.")

    # Check some values
    print(f"Loaded buffer_delay: {params.buffer_delay}")
    assert params.buffer_delay == 0.1
    print(f"Loaded buffer_size: {params.buffer_size}")
    assert params.buffer_size == 500000
    print(f"Loaded interp_hor_num (readonly): {params.interp_hor_num}") # Access readonly
    assert params.interp_hor_num == 3

    print("\n--- Testing DynObjFilterParams modification ---")
    # Modify a writable value
    params.buffer_delay = 7
    print(f"Modified buffer_delay to: {params.buffer_delay}")
    assert params.buffer_delay == 7

    # Try modifying a readonly value (should fail if bindings are correct)
    try:
            params.interp_hor_num = 10 # This should raise an AttributeError
            print("Error: Was able to modify readonly attribute 'interp_hor_num'!")
    except AttributeError:
            print("Correctly prevented modification of readonly attribute 'interp_hor_num'.")
except RuntimeError as e:
    print(f"RuntimeError during config loading: {e}")
except Exception as e:
    print(f"Unexpected error during config loading test: {e}")

def test_enums():
    """Tests accessing the exposed enums."""
    print("\n--- Testing Enums ---")
    try:
        print(f"Accessing DynObjFlg.STATIC: {mdet.DynObjFlg.STATIC}")
        print(f"Accessing DynObjFlg.CASE1: {mdet.DynObjFlg.CASE1}")
        print(f"Accessing DynObjFlg.INVALID: {mdet.DynObjFlg.INVALID}")
        # You can optionally check its integer value if needed, though using the member itself is preferred
        # print(f"Value of DynObjFlg.STATIC: {int(mdet.DynObjFlg.STATIC)}")

        print(f"Accessing InterpolationNeighborType.STATIC_ONLY: {mdet.InterpolationNeighborType.STATIC_ONLY}")
        print(f"Accessing InterpolationStatus.SUCCESS: {mdet.InterpolationStatus.SUCCESS}")
        print("Enum access successful.")
    except AttributeError as e:
        print(f"Error accessing enum: {e}")

def test_structs():
    """Tests creating and accessing the exposed structs."""
    print("\n--- Testing Structs ---")
    try:
        # Test DynObjFilterParams instantiation and access
        params = mdet.DynObjFilterParams()
        print("Created default DynObjFilterParams instance.")
        print(f"Default buffer_delay: {params.buffer_delay}") # Access default value
        params.buffer_delay = 3 # Modify value
        print(f"Set buffer_delay to: {params.buffer_delay}")
        assert params.buffer_delay == 3

        # Test InterpolationResult instantiation and access
        result = mdet.InterpolationResult()
        print("Created default InterpolationResult instance.")
        print(f"Default status: {result.status}")
        print(f"Default depth: {result.depth}")
        result.status = mdet.InterpolationStatus.SUCCESS
        result.depth = 15.5
        print(f"Set status to: {result.status}")
        print(f"Set depth to: {result.depth}")
        assert result.status == mdet.InterpolationStatus.SUCCESS
        assert abs(result.depth - 15.5) < 1e-6

        print("Struct access successful.")
    except Exception as e:
        print(f"Error during struct testing: {e}")

test_enums()
test_structs()

# --- Next Steps: Bind more and test ---
# TODO: Bind point_soph, DepthMap
# TODO: Create sample point_soph, DepthMap in Python (might be tricky)
# TODO: Bind utility functions (isPointInvalid, SphericalProjection etc.)
# TODO: Create sample NumPy points/poses
# TODO: Call bound utility functions with sample data

print("\nBasic binding tests completed.")