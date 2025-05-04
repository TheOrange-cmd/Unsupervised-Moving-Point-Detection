# PLAN

	✅ Core Structure & Buffering: Implement basic DynObjFilter class, addScan to buffer ScanFrames, processBufferedFrames to sequentially identify frames by seq_id, updateDepthMaps skeleton for map list management (creation/rotation based on time). Python tests verify map count and processed sequence ID.

	🅿️ Point Processing & Map Population:
		In processBufferedFrames: Inside the loop processing a ScanFrame:
			Iterate through the raw points (ScanFrame::cloud_ptr).
			Apply initial filtering (isPointInvalid, isSelfPoint).
			For valid points, create std::shared_ptr<point_soph> (ensure coordinates are appropriate, e.g., world frame).
			Collect these valid pointers into a std::vector<std::shared_ptr<point_soph>> processed_points.
		In updateDepthMaps:
			Receive the non-empty processed_points vector.
			For each point_soph in the vector:
				Project it into the spherical grid of the latest DepthMap.
				Perform grid cell validity checks.
				Add the point_soph pointer to the correct depth_map[row][col] vector.
				(Optional for now) Update map statistics.
		Test (Python):
			Add C++ getters (e.g., get_map_total_point_count(map_index)) and bind them.
			Call add_scan with multiple frames.
			Verify that the point counts in the created maps are non-zero and reasonable.
			(Later) Add more detailed tests for projection and grid indexing if issues arise.

	➡️ Implement checkAppearingPoint (CASE1):
		Implement ConsistencyChecks::checkMapConsistency (likely needed first).
		Implement DynObjFilter::checkAppearingPoint logic using checkMapConsistency.
		In processBufferedFrames' point loop, after initial filtering, call checkAppearingPoint. If true, set p_soph->dyn = DynObjLabel::APPEARING and continue to the next point.
		Test (Python): Add label getters. Use sequences with appearing objects. Verify APPEARING labels.

	➡️ Implement checkOccludingPoint (CASE2 - Initial):
		Implement ConsistencyChecks::findOcclusionRelationshipInMap.
		Implement DynObjFilter::checkOccludingPoint logic (using checkMapConsistency and findOcclusionRelationshipInMap, without velocity check for now).
		In processBufferedFrames' point loop, if not APPEARING, call checkOccludingPoint. If true, tentatively set p_soph->dyn = DynObjLabel::OCCLUDING (or maybe UNCERTAIN until velocity check) and store occlusion info (occu_index, is_occu_index).
		Test (Python): Use sequences with occlusion. Verify tentative labels and potentially inspect stored occlusion indices via getters.

	➡️ Implement checkDisoccludedPoint (CASE3 - Initial):
		Implement DynObjFilter::checkDisoccludedPoint logic (using checkMapConsistency and findOcclusionRelationshipInMap, without velocity check for now).
		In processBufferedFrames' point loop, if not APPEARING or OCCLUDING, call checkDisoccludedPoint. If true, tentatively set p_soph->dyn = DynObjLabel::DISOCCLUDED (or UNCERTAIN) and store relationship info. If false, set p_soph->dyn = DynObjLabel::STATIC.
		Test (Python): Use sequences with disocclusion. Verify tentative labels/STATIC labels.

	➡️ Implement Velocity/Acceleration Checks (Mandatory):
		Implement ConsistencyChecks::checkAccelerationLimit (or similar function).
		In processBufferedFrames, after the initial CASE2/CASE3 checks have potentially set occu_index/is_occu_index, add logic to:
			Trace back the occlusion history using the stored indices through previous DepthMaps in depth_map_list_.
			Call checkAccelerationLimit based on the traced points/times.
			Finalize the label: If acceleration check fails for a tentative OCCLUDING or DISOCCLUDED point, potentially revert it to STATIC or UNCERTAIN.
		Test (Python): Use specific sequences designed to test velocity limits. Verify final OCCLUDING, DISOCCLUDED, and STATIC labels are correct.
		
#KEY
	✅: Done
	🅿️: In Progress / Next Step
	➡️: Future Step