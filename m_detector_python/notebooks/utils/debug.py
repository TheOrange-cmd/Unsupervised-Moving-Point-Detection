
# def debug_box_point_labeling_for_sweep(
#     nusc,
#     scene_name,
#     sweep_index,
#     instance_token_to_debug,
#     point_labels_base_dir, # For context, not strictly used for re-evaluation here
#     downsample_visualization_points=1,
#     point_size=0.05
# ):
#     """
#     Provides a detailed visual debug for point labeling of a single instance box
#     in a specific sweep. It shows:
#     1. Global plot: OBB, its AABB, points passing AABB, points passing OBB.
#     2. Local plot: Points (that passed AABB) transformed to box's local frame,
#                    and the local axis-aligned box.

#     Args:
#         nusc: NuScenes API instance.
#         scene_name (str): The name of the scene.
#         sweep_index (int): The 0-based index of the LiDAR sweep.
#         instance_token_to_debug (str): The token of the instance whose box we want to debug.
#         point_labels_base_dir (str): Base directory of generated labels (for context).
#         downsample_visualization_points (int): Downsampling for k3d.
#         point_size (float): Point size in k3d.
#     """
#     # --- 1. Find scene and target sweep ---
#     scene_record = next((s for s in nusc.scene if s['name'] == scene_name), None)
#     if not scene_record:
#         print(f"Error: Scene '{scene_name}' not found.")
#         return None, None
#     scene_token = scene_record['token']

#     all_scene_lidar_sweeps = get_lidar_sweeps_for_interval(
#         nusc, scene_record['first_sample_token'], scene_record['last_sample_token']
#     )
#     if not (0 <= sweep_index < len(all_scene_lidar_sweeps)):
#         print(f"Error: sweep_index {sweep_index} out of range.")
#         return None, None
    
#     target_sweep_data = all_scene_lidar_sweeps[sweep_index]
#     lidar_sd_token = target_sweep_data['token']
#     print(f"Debugging Labeling: Scene '{scene_name}', Sweep Idx {sweep_index} (Token: {lidar_sd_token}), Instance '{instance_token_to_debug[:6]}...'")

#     # --- 2. Load all LiDAR points for the target sweep ---
#     points_global_full = load_lidar_points_global(nusc, lidar_sd_token, downsample_factor=1)
#     if points_global_full.shape[0] == 0:
#         print("Error: No points in LiDAR sweep.")
#         return None, None
    
#     # Downsample for visualization if requested (apply to all point sets derived from this)
#     points_global = points_global_full[::downsample_visualization_points]


#     # --- 3. Get the specific interpolated/extrapolated box for the instance ---
#     boxes_for_instance_all_sweeps, _, _, _ = get_interpolated_extrapolated_boxes_for_instance(
#         nusc, instance_token_to_debug, all_scene_lidar_sweeps
#     )
#     box_object = boxes_for_instance_all_sweeps[sweep_index]

#     if box_object is None:
#         print(f"Error: No box found for instance '{instance_token_to_debug}' at sweep index {sweep_index}.")
#         return None, None
    
#     print(f"Target Box Details: Center={box_object.center}, WLH={box_object.wlh}, Orientation(q)={box_object.orientation.elements}")

#     # --- 4. Perform Broad-Phase Culling (AABB check) ---
#     obb_corners_global = box_object.corners().T  # (8,3)
#     aabb_min_global = np.min(obb_corners_global, axis=0)
#     aabb_max_global = np.max(obb_corners_global, axis=0)
#     print(f"  Broad-Phase: OBB's Global AABB Min={aabb_min_global}, Max={aabb_max_global}")

#     mask_in_aabb_x = (points_global[:, 0] >= aabb_min_global[0]) & (points_global[:, 0] <= aabb_max_global[0])
#     mask_in_aabb_y = (points_global[:, 1] >= aabb_min_global[1]) & (points_global[:, 1] <= aabb_max_global[1])
#     mask_in_aabb_z = (points_global[:, 2] >= aabb_min_global[2]) & (points_global[:, 2] <= aabb_max_global[2])
#     mask_in_aabb = mask_in_aabb_x & mask_in_aabb_y & mask_in_aabb_z
    
#     candidate_indices_global_vis = np.where(mask_in_aabb)[0] # Indices relative to 'points_global' (downsampled)
#     candidate_points_global_vis = points_global[candidate_indices_global_vis]
#     print(f"  Broad-Phase: {len(candidate_indices_global_vis)} (downsampled) points passed AABB check out of {len(points_global)}.")

#     # --- 5. Perform Narrow-Phase Check (Precise OBB) on candidates ---
#     final_indices_in_obb_relative_to_candidates = np.array([], dtype=int)
#     if candidate_points_global_vis.shape[0] > 0:
#         mask_of_candidates_in_obb = _get_points_in_box_mask_global_coords(
#             candidate_points_global_vis, box_object, debug_instance_token=instance_token_to_debug # Pass candidate points
#         )
#         final_indices_in_obb_relative_to_candidates = np.where(mask_of_candidates_in_obb)[0]
    
#     # Map these indices back to the original 'points_global' (downsampled) array
#     final_indices_in_obb_global_vis = candidate_indices_global_vis[final_indices_in_obb_relative_to_candidates]
#     points_in_obb_vis = points_global[final_indices_in_obb_global_vis]
#     print(f"  Narrow-Phase: {len(final_indices_in_obb_global_vis)} (downsampled) points passed OBB check.")


#     # --- Plot 1: Global Coordinates Debug ---
#     plot_global = k3d.plot(name=f"Global Debug: {scene_name[:10]}:{sweep_index} Inst:{instance_token_to_debug[:6]}")
    
#     # Plot all points (grey)
#     plot_global += k3d.points(points_global.astype(np.float32), color=0xAAAAAA, point_size=point_size, name="All Points")
    
#     # Plot points that passed AABB (yellow)
#     if candidate_points_global_vis.shape[0] > 0:
#         plot_global += k3d.points(candidate_points_global_vis.astype(np.float32), color=0xFFFF00, point_size=point_size*1.2, name="In AABB")

#     # Plot points that passed OBB (green)
#     if points_in_obb_vis.shape[0] > 0:
#         plot_global += k3d.points(points_in_obb_vis.astype(np.float32), color=0x00FF00, point_size=point_size*1.4, name="In OBB (Labeled)")

#     # Plot the OBB (blue)
#     box_edges = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]
#     for start_idx, end_idx in box_edges:
#         segment = obb_corners_global[[start_idx, end_idx], :].astype(np.float32)
#         plot_global += k3d.line(segment, shader='simple', color=0x0000FF, width=0.07, name="Target OBB")

#     # Plot the AABB (red)
#     aabb_corners = np.array([
#         [aabb_min_global[0], aabb_min_global[1], aabb_min_global[2]],
#         [aabb_max_global[0], aabb_min_global[1], aabb_min_global[2]],
#         [aabb_max_global[0], aabb_max_global[1], aabb_min_global[2]],
#         [aabb_min_global[0], aabb_max_global[1], aabb_min_global[2]],
#         [aabb_min_global[0], aabb_min_global[1], aabb_max_global[2]],
#         [aabb_max_global[0], aabb_min_global[1], aabb_max_global[2]],
#         [aabb_max_global[0], aabb_max_global[1], aabb_max_global[2]],
#         [aabb_min_global[0], aabb_max_global[1], aabb_max_global[2]],
#     ])
#     for start_idx, end_idx in box_edges: # Use same edge definitions
#         segment = aabb_corners[[start_idx, end_idx], :].astype(np.float32)
#         plot_global += k3d.line(segment, shader='simple', color=0xFF0000, width=0.03, name="OBB's AABB")
#     plot_global.camera_auto_fit = True


#     # --- Plot 2: Box Local Coordinates Debug ---
#     plot_local = k3d.plot(name=f"Local Debug: {scene_name[:10]}:{sweep_index} Inst:{instance_token_to_debug[:6]}")
    
#     # Transform candidate points (that passed AABB) to box's local frame
#     local_points_to_check = np.array([[]]).reshape(0,3) # Default empty
#     if candidate_points_global_vis.shape[0] > 0:
#         points_translated = candidate_points_global_vis - box_object.center
#         local_points_to_check = points_translated @ box_object.rotation_matrix.T

#         # Colors for local points: green if they passed OBB, red otherwise (among candidates)
#         local_point_colors = np.full(len(candidate_points_global_vis), 0xFF0000, dtype=np.uint32) # Default red (failed OBB)
#         # Create a boolean mask: True for points in 'candidate_points_global_vis' that are also in 'points_in_obb_vis'
#         # This requires careful index mapping or direct re-evaluation of the mask_of_candidates_in_obb
#         # For simplicity, we re-use mask_of_candidates_in_obb which is already aligned with candidate_points_global_vis
#         if 'mask_of_candidates_in_obb' in locals() and mask_of_candidates_in_obb.any():
#              local_point_colors[mask_of_candidates_in_obb] = 0x00FF00 # Green (passed OBB)
        
#         plot_local += k3d.points(local_points_to_check.astype(np.float32), 
#                                  colors=local_point_colors, 
#                                  point_size=point_size, name="Candidate Points (Local)")

#     # Draw the axis-aligned local box centered at origin
#     w, l, h = box_object.wlh
#     half_w, half_l, half_h = w/2, l/2, h/2
#     local_box_corners = np.array([
#         [-half_w, -half_l, -half_h], [ half_w, -half_l, -half_h], [ half_w,  half_l, -half_h], [-half_w,  half_l, -half_h],
#         [-half_w, -half_l,  half_h], [ half_w, -half_l,  half_h], [ half_w,  half_l,  half_h], [-half_w,  half_l,  half_h]
#     ])
#     for start_idx, end_idx in box_edges:
#         segment = local_box_corners[[start_idx, end_idx], :].astype(np.float32)
#         plot_local += k3d.line(segment, shader='simple', color=0x0000FF, width=0.05, name="Local Box Frame")
#     plot_local.camera_auto_fit = True
    
#     print("\nDebug Plots Generated. Inspect them carefully.")
#     print("Global Plot: All points (grey), In AABB (yellow), In OBB (green), Target OBB (blue), OBB's AABB (red).")
#     print("Local Plot: Candidate points transformed to box local frame. Green=passed OBB, Red=failed OBB. Local box (blue).")

#     return plot_global, plot_local

# def get_camera_data_for_sweep(nusc, lidar_sd_token, camera_name='CAM_FRONT'):
#     """
#     Checks if the LiDAR sweep is a keyframe and retrieves data for a specified camera.

#     Args:
#         nusc: NuScenes API instance.
#         lidar_sd_token (str): Token of the LIDAR_TOP sample_data.
#         camera_name (str): Name of the camera (e.g., 'CAM_FRONT').

#     Returns:
#         tuple: (image_path, intrinsics, global_to_cam_transform, image_size) or (None, None, None, None) if not a keyframe
#                or camera data not found.
#                global_to_cam_transform: Matrix that transforms points from global to camera sensor frame.
#     """
#     lidar_sd_rec = nusc.get('sample_data', lidar_sd_token)
#     if not lidar_sd_rec['is_key_frame']:
#         # print(f"LiDAR sweep {lidar_sd_token} is not a keyframe. Camera data might not be perfectly synchronized.")
#         return None, None, None, None, None

#     sample_token = lidar_sd_rec['sample_token']
#     sample_rec = nusc.get('sample', sample_token)

#     if camera_name not in sample_rec['data']:
#         print(f"Camera {camera_name} not found in sample {sample_token}.")
#         return None, None, None, None, None

#     cam_sd_token = sample_rec['data'][camera_name]
#     cam_sd_rec = nusc.get('sample_data', cam_sd_token)
#     image_path = os.path.join(nusc.dataroot, cam_sd_rec['filename'])
    
#     cam_cs_rec = nusc.get('calibrated_sensor', cam_sd_rec['calibrated_sensor_token'])
#     cam_intrinsics = np.array(cam_cs_rec['camera_intrinsic'])
    
#     # Transformation: Global -> Ego -> Sensor
#     ego_pose_rec = nusc.get('ego_pose', cam_sd_rec['ego_pose_token'])

#     # Matrix from ego to global
#     ego_to_global_trans = np.array(ego_pose_rec['translation'])
#     ego_to_global_rot = Quaternion(ego_pose_rec['rotation'])
    
#     # Matrix from sensor to ego
#     sensor_to_ego_trans = np.array(cam_cs_rec['translation'])
#     sensor_to_ego_rot = Quaternion(cam_cs_rec['rotation'])

#     # Homogeneous transformation matrices
#     global_from_ego_tf = transform_matrix(ego_pose_rec['translation'], Quaternion(ego_pose_rec['rotation']), S=np.eye(4))
#     ego_from_sensor_tf = transform_matrix(cam_cs_rec['translation'], Quaternion(cam_cs_rec['rotation']), S=np.eye(4))

#     # To get Global -> Sensor, we need inv(Sensor -> Ego) @ inv(Ego -> Global)
#     # Or, Sensor <- Ego <- Global
#     sensor_from_ego_tf = np.linalg.inv(ego_from_sensor_tf)
#     ego_from_global_tf = np.linalg.inv(global_from_ego_tf)

#     global_to_sensor_tf = sensor_from_ego_tf @ ego_from_global_tf
    
#     image_size = (cam_sd_rec['width'], cam_sd_rec['height'])

#     return image_path, cam_intrinsics, global_to_sensor_tf, image_size, cam_sd_token


# def project_points_to_image_plane(points_global, global_to_cam_sensor_tf, cam_intrinsics):
#     """
#     Projects 3D points from global frame to 2D image coordinates.

#     Args:
#         points_global (np.ndarray): Nx3 array of points in global coordinates.
#         global_to_cam_sensor_tf (np.ndarray): 4x4 matrix transforming global to camera sensor frame.
#         cam_intrinsics (np.ndarray): 3x3 camera intrinsic matrix.

#     Returns:
#         np.ndarray: Nx2 array of (u,v) image coordinates.
#         np.ndarray: N array of depths (z-coordinate in camera frame).
#     """
#     if points_global.ndim == 1:
#         points_global = points_global.reshape(1, -1)
#     num_points = points_global.shape[0]

#     # Transform points to camera sensor frame
#     points_global_hom = np.hstack((points_global, np.ones((num_points, 1)))) # Nx4
#     points_cam_sensor_hom = (global_to_cam_sensor_tf @ points_global_hom.T).T # Nx4

#     # Points in camera sensor frame (non-homogeneous)
#     points_cam_sensor = points_cam_sensor_hom[:, :3] # Nx3
#     depths = points_cam_sensor[:, 2].copy() # Z-coordinate in camera frame

#     # Project to image plane using intrinsics
#     # view_points expects points as 3xN
#     points_img_coords_hom = view_points(points_cam_sensor.T, cam_intrinsics, normalize=True) # 3xN

#     # Transpose to get Nx3, then take first two for (u,v)
#     points_img_coords = points_img_coords_hom.T[:, :2] # Nx2

#     return points_img_coords, depths


# def display_image_with_projected_points(
#     image_path, points_img_coords, depths, image_size,
#     point_colors=None, point_size=5, circle_radius=5
# ):
#     """
#     Displays an image with projected points marked.

#     Args:
#         image_path (str): Path to the camera image.
#         points_img_coords (np.ndarray): Nx2 array of (u,v) coordinates.
#         depths (np.ndarray): N array of depths for filtering points behind camera.
#         image_size (tuple): (width, height) of the image.
#         point_colors (list/array, optional): Colors for each point.
#         point_size (int, optional): Size of the scatter plot points.
#         circle_radius (int, optional): Radius for drawing circles (alternative to scatter).
#     """
#     try:
#         img = Image.open(image_path)
#     except FileNotFoundError:
#         print(f"Error: Image not found at {image_path}")
#         return

#     img_width, img_height = image_size

#     plt.figure(figsize=(12, int(12 * img_height / img_width)))
#     plt.imshow(img)
    
#     valid_mask = (points_img_coords[:, 0] >= 0) & (points_img_coords[:, 0] < img_width) & \
#                  (points_img_coords[:, 1] >= 0) & (points_img_coords[:, 1] < img_height) & \
#                  (depths > 0) # Points must be in front of camera

#     valid_points_img = points_img_coords[valid_mask]
    
#     if valid_points_img.shape[0] > 0:
#         if point_colors is not None:
#             valid_colors = np.array(point_colors)[valid_mask]
#             plt.scatter(valid_points_img[:, 0], valid_points_img[:, 1], s=point_size**2, c=valid_colors, edgecolors='black', linewidths=0.5)
#         else:
#             plt.scatter(valid_points_img[:, 0], valid_points_img[:, 1], s=point_size**2, c='red', edgecolors='black', linewidths=0.5)
        
#         # Alternative: draw circles
#         # for i in range(valid_points_img.shape[0]):
#         #     color = valid_colors[i] if point_colors is not None else 'red'
#         #     circ = plt.Circle((valid_points_img[i, 0], valid_points_img[i, 1]), circle_radius, color=color, fill=False, linewidth=1.5)
#         #     plt.gca().add_artist(circ)

#     plt.title(f"Camera: {os.path.basename(image_path)}")
#     plt.axis('off')
#     plt.show()

# def visualize_lidar_and_boxes_on_camera_image(
#     nusc,
#     scene_name,
#     target_sweep_index, # The sweep index we want to visualize (should be a keyframe)
#     camera_name='CAM_FRONT',
#     points_dist_max=50.0, # Max distance for LiDAR points to render
#     dot_size=2,
#     box_line_width=2,
#     specific_instance_token_to_highlight=None, # Optional: token of an instance to highlight
#     show_interpolated_box_if_available=False, # If true, and specific_instance_token is given, try to show our interpolated box
#     all_scene_lidar_sweeps_for_interp=None # Needed if show_interpolated_box_if_available is True
# ):
#     """
#     Visualizes LiDAR points and 3D annotation boxes projected onto a camera image
#     for a specific keyframe sweep.

#     Args:
#         nusc: NuScenes API instance.
#         scene_name (str): Name of the scene.
#         target_sweep_index (int): Index of the LIDAR_TOP sweep in the scene's timeline.
#                                  This sweep MUST correspond to a keyframe.
#         camera_name (str): Name of the camera channel (e.g., 'CAM_FRONT').
#         points_dist_max (float): Maximum distance for LiDAR points to be visualized.
#         dot_size (int): Size of the LiDAR points in the plot.
#         box_line_width (int): Width of the lines for drawing boxes.
#         specific_instance_token_to_highlight (str, optional): If provided, this instance's box
#                                                              will be highlighted.
#         show_interpolated_box_if_available (bool): If True and specific_instance_token_to_highlight
#                                                    is provided, also try to plot our interpolated box.
#         all_scene_lidar_sweeps_for_interp (list, optional): List of all lidar sweeps in the scene,
#                                                             required if showing interpolated box.
#     """
#     # --- 1. Get Scene and Target LiDAR Sweep Data ---
#     scene_rec = next((s for s in nusc.scene if s['name'] == scene_name), None)
#     if not scene_rec:
#         print(f"Error: Scene '{scene_name}' not found.")
#         return

#     # If all_scene_lidar_sweeps_for_interp is not provided, fetch them.
#     # This list is assumed to be ordered by time.
#     if all_scene_lidar_sweeps_for_interp is None:
#          all_scene_lidar_sweeps_for_interp = get_lidar_sweeps_for_interval(
#             nusc, scene_rec['first_sample_token'], scene_rec['last_sample_token']
#         )

#     if not (0 <= target_sweep_index < len(all_scene_lidar_sweeps_for_interp)):
#         print(f"Error: target_sweep_index {target_sweep_index} is out of range "
#               f"(0 to {len(all_scene_lidar_sweeps_for_interp) - 1}).")
#         return
    
#     lidar_sd_rec = all_scene_lidar_sweeps_for_interp[target_sweep_index]
#     lidar_sd_token = lidar_sd_rec['token']

#     if not lidar_sd_rec['is_key_frame']:
#         print(f"Error: Sweep index {target_sweep_index} (Token: {lidar_sd_token}) is NOT a keyframe. "
#               "This function requires a keyframe sweep for synchronized annotations and camera data.")
#         return

#     sample_token = lidar_sd_rec['sample_token']
#     sample_rec = nusc.get('sample', sample_token)

#     # --- 2. Get Camera Data (Intrinsics, Extrinsics, Image Path) ---
#     cam_sd_token = sample_rec['data'].get(camera_name)
#     if not cam_sd_token:
#         print(f"Error: Camera '{camera_name}' not found in sample {sample_token}.")
#         return
    
#     cam_sd_rec = nusc.get('sample_data', cam_sd_token)
#     cam_cs_rec = nusc.get('calibrated_sensor', cam_sd_rec['calibrated_sensor_token'])
#     cam_img_path = nusc.get_sample_data_path(cam_sd_token) # Use official way to get path
#     cam_intrinsics = np.array(cam_cs_rec['camera_intrinsic'])
#     img_height, img_width = cam_sd_rec['height'], cam_sd_rec['width']

#     # Transformation: Global to Camera Sensor Frame (at camera timestamp)
#     cam_ego_pose_rec = nusc.get('ego_pose', cam_sd_rec['ego_pose_token'])
    
#     # Rotation from ego to global at camera time
#     ego_to_global_rot_cam = Quaternion(cam_ego_pose_rec['rotation'])
#     # Translation from ego to global at camera time
#     ego_to_global_trans_cam = np.array(cam_ego_pose_rec['translation'])

#     # Rotation from camera sensor to ego at camera time
#     sensor_to_ego_rot_cam = Quaternion(cam_cs_rec['rotation'])
#     # Translation from camera sensor to ego at camera time
#     sensor_to_ego_trans_cam = np.array(cam_cs_rec['translation'])

#     # --- 3. Load LiDAR Points for the Target Sweep (Global Coords) ---
#     points_global = load_lidar_points_global(nusc, lidar_sd_token, downsample_factor=1) # Nx3 array
    
#     # Project LiDAR points from Global to Image Pixels
#     # Step 1: Global to Ego frame at camera timestamp
#     points_ego_cam_time = points_global - ego_to_global_trans_cam # Translate
#     points_ego_cam_time = np.dot(ego_to_global_rot_cam.inverse.rotation_matrix, points_ego_cam_time.T).T # Rotate

#     # Step 2: Ego frame at camera timestamp to Camera Sensor frame at camera timestamp
#     points_sensor_cam_time = points_ego_cam_time - sensor_to_ego_trans_cam # Translate
#     points_sensor_cam_time = np.dot(sensor_to_ego_rot_cam.inverse.rotation_matrix, points_sensor_cam_time.T).T # Rotate
    
#     # Step 3: Camera Sensor frame to Image Pixels
#     points_img_coords_hom = view_points(points_sensor_cam_time.T, cam_intrinsics, normalize=True) # 3xN
#     projected_lidar_pixels = points_img_coords_hom[:2, :].T # Nx2 (u,v)
#     lidar_depths_in_cam = points_sensor_cam_time[:, 2] # Z-coordinate in camera frame

#     # --- 4. Get Annotations for this Sample (Keyframe) ---
#     anns_tokens = sample_rec['anns']
    
#     # --- 5. Plotting ---
#     try:
#         img = Image.open(cam_img_path)
#     except FileNotFoundError:
#         print(f"Error: Image not found at {cam_img_path}")
#         return

#     fig, ax = plt.subplots(1, 1, figsize=(16, int(16 * img_height / img_width))) # Increased figure size
#     ax.imshow(img)

#     # Plot LiDAR points
#     valid_lidar_mask = (projected_lidar_pixels[:, 0] >= 0) & (projected_lidar_pixels[:, 0] < img_width -1) & \
#                        (projected_lidar_pixels[:, 1] >= 0) & (projected_lidar_pixels[:, 1] < img_height -1) & \
#                        (lidar_depths_in_cam > 0.1) & (lidar_depths_in_cam < points_dist_max) # Min depth 0.1m
    
#     ax.scatter(projected_lidar_pixels[valid_lidar_mask, 0], 
#                projected_lidar_pixels[valid_lidar_mask, 1], 
#                c=lidar_depths_in_cam[valid_lidar_mask], 
#                s=dot_size, cmap='viridis_r', alpha=0.6, edgecolors='none') # viridis_r: closer is brighter

#     # Plot Annotation Boxes (Ground Truth from keyframe)
#     edges = np.array([[0,1],[1,2],[2,3],[3,0],[0,4],[1,5],[2,6],[3,7],[4,5],[5,6],[6,7],[7,4]]) # Box edges

#     for ann_token in anns_tokens:
#         ann_rec = nusc.get('sample_annotation', ann_token)
#         box_global_gt = NuScenesDataClassesBox(ann_rec['translation'], ann_rec['size'], Quaternion(ann_rec['rotation']))

#         # Project GT box to image
#         # Step 1: Global to Ego frame at camera timestamp
#         corners_ego_cam_time = box_global_gt.corners().T - ego_to_global_trans_cam
#         corners_ego_cam_time = np.dot(ego_to_global_rot_cam.inverse.rotation_matrix, corners_ego_cam_time.T).T

#         # Step 2: Ego frame at camera timestamp to Camera Sensor frame
#         corners_sensor_cam_time = corners_ego_cam_time - sensor_to_ego_trans_cam
#         corners_sensor_cam_time = np.dot(sensor_to_ego_rot_cam.inverse.rotation_matrix, corners_sensor_cam_time.T).T
        
#         # Only draw if box is somewhat in front and not too far
#         if np.mean(corners_sensor_cam_time[:, 2]) > 0.5 and np.mean(corners_sensor_cam_time[:, 2]) < 100:
#              # Check visibility using NuScenes method (optional, can be slow for many boxes)
#             # box_visibility = nusc.explorer.box_visibility(box_global_gt, cam_intrinsics, sensor_to_ego_trans_cam, sensor_to_ego_rot_cam, ego_to_global_trans_cam, ego_to_global_rot_cam)
#             # if box_visibility == BoxVisibility.NONE: # Or PARTIAL, ALL
#             #     continue

#             if np.all(corners_sensor_cam_time[:, 2] > 0.1): # All corners must be in front
#                 corners_img_hom = view_points(corners_sensor_cam_time.T, cam_intrinsics, normalize=True)
#                 corners_img_pixels = corners_img_hom[:2, :].T # 8x2 (u,v)

#                 box_color = 'lime' if ann_rec['instance_token'] == specific_instance_token_to_highlight else 'red'
                
#                 for i, j in edges:
#                     ax.plot([corners_img_pixels[i, 0], corners_img_pixels[j, 0]],
#                             [corners_img_pixels[i, 1], corners_img_pixels[j, 1]],
#                             color=box_color, linewidth=box_line_width, alpha=0.7)

#     # --- Optionally, plot our interpolated/extrapolated box for the highlighted instance ---
#     if show_interpolated_box_if_available and specific_instance_token_to_highlight and \
#        all_scene_lidar_sweeps_for_interp is not None:
        
#         boxes_for_instance_all_sweeps, _, _, _ = get_interpolated_extrapolated_boxes_for_instance(
#             nusc, specific_instance_token_to_highlight, all_scene_lidar_sweeps_for_interp
#         )
#         our_box_global = boxes_for_instance_all_sweeps[target_sweep_index]

#         if our_box_global is not None:
#             # Project our_box_global to image (same steps as GT box)
#             our_corners_ego_cam_time = our_box_global.corners().T - ego_to_global_trans_cam
#             our_corners_ego_cam_time = np.dot(ego_to_global_rot_cam.inverse.rotation_matrix, our_corners_ego_cam_time.T).T
#             our_corners_sensor_cam_time = our_corners_ego_cam_time - sensor_to_ego_trans_cam
#             our_corners_sensor_cam_time = np.dot(sensor_to_ego_rot_cam.inverse.rotation_matrix, our_corners_sensor_cam_time.T).T

#             if np.mean(our_corners_sensor_cam_time[:, 2]) > 0.5 and np.mean(our_corners_sensor_cam_time[:, 2]) < 100:
#                 if np.all(our_corners_sensor_cam_time[:, 2] > 0.1):
#                     our_corners_img_hom = view_points(our_corners_sensor_cam_time.T, cam_intrinsics, normalize=True)
#                     our_corners_img_pixels = our_corners_img_hom[:2, :].T
                    
#                     for i, j in edges:
#                         ax.plot([our_corners_img_pixels[i, 0], our_corners_img_pixels[j, 0]],
#                                 [our_corners_img_pixels[i, 1], our_corners_img_pixels[j, 1]],
#                                 color='cyan', linewidth=box_line_width, linestyle='--', alpha=0.8)

#     ax.set_xlim(0, img_width)
#     ax.set_ylim(img_height, 0) 
#     ax.axis('off')
#     ax.set_title(f"{scene_name} - Sweep {target_sweep_index} ({lidar_sd_token[:6]}) - Cam: {camera_name}\n"
#                  f"GT Boxes: Red/Lime (Lime if instance '{str(specific_instance_token_to_highlight)[:6]}...')\n"
#                  f"Our Interp Box (if shown): Cyan Dashed", fontsize=10)
#     plt.tight_layout()
#     plt.show()
