Validation Notes

6 images used in validation
- folder C:\development\Python\projects\pose_estimation\PoseEstimation\bicycleabout\validation\images
- no background
- different side views, right left
- some bicycle only, some with rider
- TRek 520 has wheel 29"; cervelo 27; specialized:30; novara: 27


test specific items
- dy_nobgnd_20260528_145141.png, One thing this test should specifically catch: knee angles are allowed to be computed as raw measurements, but the feedback logic should not judge them against target fit ranges unless the crank/pedal position is valid. So this image is perfect for testing the new “computed but not fit-evaluated” behavior.
- dy_straight_centered_20260528_182744.png, This is another good bike + rider validation case, but it also exposes an important issue: the first /analyze-bike-geometry pass had cm_per_px: None, so feedback returned empty because all distances were pixel-only. Later /annotate successfully computed scale from the front wheel at about 0.2192259 cm/px, and the later bike-landmark metrics had real cm/in values. That makes this a useful regression case for scale availability / recalculation consistency.
      [BIKE GEOMETRY] cm_per_px: None
      ...
      [SERVER DEBUG] /feedback Returned feedback: []
      ...
      /annotate: AFTER compute_conversion_factor(...cm_per_px=0.21922589844175275...)
      ...
      Legacy bike metric refresh endpoint returned 404 before the client used /analyze-bike-geometry consistently.
- 
