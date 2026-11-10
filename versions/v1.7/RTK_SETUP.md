# RTK setup for this rig

Start indoors with the D455 and the room scan. You do not need GPS for that workflow.

Your ZED-F9P is a GNSS receiver; the GENX antenna receives satellite signals. RTK also needs correction data. A fixed base receiver, or an internet correction service delivered through an NTRIP client, supplies that data to your rover. The receiver reports RTK float while resolving ambiguities, and RTK fixed when it has a fixed solution. See the [u-blox integration manual](https://content.u-blox.com/sites/default/files/ZED-F9P_IntegrationManual_UBX-18010802.pdf).

1. Mount the receiver antenna rigidly to the camera rig and test outdoors with a clear sky view. Keep `gnss.calibrated: false`.
2. Identify the serial connection with `python3 tracking_preflight.py`. Prefer a stable `/dev/serial/by-id/...` path. Enter it in the dashboard’s **Setup** tab and press **Connect GNSS**.
3. Arrange a compatible correction source: a local fixed base, or an NTRIP service covering your location. Obtain its connection details from the operator. This project does not include an NTRIP client or send corrections to the receiver.
4. Configure the receiver using the tools and documentation appropriate to its firmware. Enable NMEA **GGA and GST** on the output used by this bridge. GGA supplies fix type, satellite count and correction age; GST supplies error estimates. The bridge requires matching message epochs and valid checksums.
5. Verify the dashboard shows **RTK fixed**, recent correction age and small horizontal/vertical error estimates. Fixed status alone is insufficient.
6. Survey a reference point and its coordinates in the displayed scan world. Enter its latitude, longitude and **ellipsoidal** height as `origin_lla`, and its displayed ROS coordinates in metres as `origin_world_m`.
7. Measure `enu_yaw_deg`: the rotation about world Z that maps East/North/Up to the displayed ROS world. A positive angle turns East toward positive world Y. The scan must first be aligned to gravity; one yaw cannot correct a tilted world.
8. Measure `antenna_offset_body_m` from the D455 tracking origin to the antenna phase centre: X forward, Y left, Z up. This is separate from the taking lens’s nodal offset.
9. Measure receiver/serial latency and enter `latency_s`. This implementation pairs RTK with the camera’s host reception history; it is not a hardware timestamp synchronization system.
10. Compare stationary positions and known movements, then set `calibrated: true` in the active configuration and restart.

A single stationary antenna supplies a position, not camera pan, tilt and roll. RTK here **checks** camera-derived position; it does not replace the camera trajectory or estimate heading. If scan validation disappears, fresh calibrated RTK agreement can support continued camera tracking. If camera pose disappears, output holds.

The default GNSS gates require fixed RTK, at least 8 satellites, HDOP at most 2, corrections no older than 2 seconds, horizontal error at most 5 cm, vertical error at most 10 cm, and a camera/antenna residual at most 20 cm. These are configurable engineering thresholds, not demonstrated tracking accuracy.

The receiver reader opens its serial port for input and never sends configuration or corrections. Use a separate receiver interface for RTCM delivery, or an external multiplexer; two programs should not compete for the same serial port. Do not enable fusion solely because the dashboard says “RTK fixed.”

The antenna’s advertised frequency bands do not add receiver capabilities. Check the exact F9P variant and firmware before choosing correction formats or services.
