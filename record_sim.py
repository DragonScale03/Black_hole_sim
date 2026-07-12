import taichi as ti
import numpy as np
import cv2
import main

print("Recording simulation to blackhole_sim.mp4...")

main.update_camera()
main.init_particles()
main.init_stars()

video = cv2.VideoWriter('blackhole_sim.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (main.WIN_W, main.WIN_H))

rng = np.random.default_rng(7)

# Record 300 frames (10 seconds)
for frame in range(300):
    # Slowly orbit the camera for a cinematic shot
    main.cam_phi += 0.005
    main.update_camera()
    
    main.update_particles()
    main.decay_fb()
    main.render_particles()
    main.render_stars_lensed()
    main.render_horizon()
    main.tonemap()
    
    # Grab the frame from the framebuffer
    pixels = main.fb.to_numpy()
    
    # Taichi provides (W, H, 3) RGB float32 in [0, 1]
    # OpenCV expects (H, W, 3) BGR uint8 in [0, 255]
    img = np.clip(pixels, 0, 1) * 255.0
    img = img.astype(np.uint8)
    img = np.swapaxes(img, 0, 1) 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    video.write(img)
    
    if frame % 50 == 0 and frame > 0:
        main.respawn_dead(rng, n=60)
        
video.release()
print("Successfully saved blackhole_sim.mp4")
