# RAPP Lab 04: Vision-Action Model
## Design Brief & Technical Overview

**Project Lead:** Dr. Maleen Jayasuriya  
**Collaborators:** Piumi Wijesundara, Prof. Damith Herath  
**Institution:** University of Canberra - Collaborative Robotics Lab  

---

## Executive Summary

RAPP Lab 04 develops a Vision-Action Model (VAM) using Action Chunking Transformers to enable a UR10 collaborative robot to mirror and respond to human performers in real-time improvised physical theatre. The system learns from paired skeletal tracking data (ZED 2i camera) and robot joint configurations to generate fluid, intentional-feeling robotic movements that foster complicité in human-robot ensemble performance.

**Key Innovation:** Transitioning the robot from executing pre-programmed sequences to real-time responsive improvisation while maintaining perceived agency and theatrical intentionality.

---

## Project Context

Building on RAPP Lab 03's exploration of human-robot ensemble storytelling through Meyerhold's biomechanics and Lecoq's Tréteau theatre, this iteration introduces AI-enabled improvisation. The VAM enables the robot to function as a genuine ensemble partner capable of in-the-moment response rather than scripted performance.

**Research Questions:**
1. Can VAMs function as creative dramaturgical agents capable of improvised physical theatre?
2. How does perceived agency influence human-robot complicité development?
3. Can robots exercise agency and co-authorship within ensemble-based narrative development?
4. How can physical theatre frameworks establish shared movement vocabulary for human-robot storytelling?

---

## System Architecture

### Overall Data Flow

```
Data Collection (Completed)
    ↓
[ZED 2i Camera] → Skeleton Tracking (16 keypoints, 3D)
[UR10 Robot] → Joint Angles (6 DOF)
[Freedrive Mirroring] → Performer mirrors leader using robot
    ↓
ROS2 Rosbags (Raw recordings)
    ↓
STEP 1: Docker Environment Setup
    ↓
STEP 2: Data Processing Pipeline
    ├─ Extract & Synchronize Topics
    ├─ Transform to Robot Frame
    ├─ Interactive Skeleton Selection
    └─ Export to CSV Training Data
    ↓
STEP 3: Model Development
    ├─ Prepare PyTorch Datasets
    ├─ Train Action Chunking Transformer
    ├─ Evaluate & Visualize
    └─ Save Trained Model
    ↓
STEP 4: Real-Time Inference
    ├─ ROS2 Inference Node
    ├─ Live Skeleton Input
    ├─ Predict Robot Trajectories
    └─ Execute on UR10
```

### Container Architecture

```
┌─────────────────────────────────────────────┐
│         RAPP Lab 04 Infrastructure          │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐      ┌──────────────┐   │
│  │ ZED2 Docker  │      │  VAM Docker  │   │
│  │ (Existing)   │◄────►│   (New)      │   │
│  │              │ ROS2 │              │   │
│  │ - ZED SDK    │Topic │ - ROS2       │   │
│  │ - UR Driver  │Bridge│ - PyTorch    │   │
│  │ - Tracking   │      │ - Jupyter    │   │
│  └──────────────┘      └──────────────┘   │
│                                             │
│  Shared Volumes:                            │
│  ├─ /data/rosbags/    (Raw recordings)    │
│  ├─ /data/processed/  (CSV files)         │
│  ├─ /data/models/     (Checkpoints)       │
│  └─ /data/config/     (URDF, params)      │
└─────────────────────────────────────────────┘
```

---

## Technical Specifications

### Hardware & Software Stack

**Computing:**
- GPU: RTX 5070 Ti (CUDA 12.1)
- OS: Ubuntu 22.04
- Containerization: Docker with GPU passthrough

**ROS2 Environment:**
- ROS2 Humble
- Network: Host mode for inter-container communication
- Domain ID: 0 (matching ZED2 container)

**Machine Learning:**
- Framework: PyTorch 2.1+ with CUDA
- Model: Action Chunking Transformer (ACT)
- Training: JupyterLab environment

**Robotics:**
- Robot: Universal Robots UR10
- Sensor: ZED 2i stereo camera
- Kinematics: URDF-based forward kinematics

### Data Specifications

**Input Data (From Rosbags):**
- Skeleton Tracking: 16 keypoints × 3 coordinates (48 dimensions)
- Robot State: 6 joint angles (radians)
- Frequency: Skeleton ~30Hz, Joints ~125Hz
- Coordinate Frame: Both in robot_base_link after transformation

**ROS2 Topics:**
```
/zed/zed_node/body_trk/skeletons  → Skeleton data (may contain 2 people)
/joint_states                      → UR10 joint positions
/robot_description                 → URDF (extract once)
/tf_static                         → Static transforms
```

**Training Data Format:**
- Temporal sequences: Input window (T_in=10 frames) → Output prediction (T_out=50 frames)
- Input: [skeleton_history: [10, 48], robot_history: [10, 6]]
- Output: [robot_future: [50, 6]]

### Model Architecture

**Action Chunking Transformer:**

```
Input Processing:
├─ Skeleton Encoder (Temporal Transformer)
│  ├─ 6 layers, 8 attention heads
│  ├─ Dimension: 256
│  └─ Output: Skeleton latent representation [256]
│
├─ Robot State Encoder (MLP)
│  ├─ Hidden layers: [64, 128]
│  └─ Output: Robot latent representation [128]
│
└─ Fusion Layer → Combined latent [384]

Action Decoder:
├─ Transformer Decoder (4 layers, 8 heads)
├─ Predicts entire trajectory chunk (50 frames)
└─ Output: [T_out=50, 6] joint angle predictions

Loss Function:
├─ Prediction Loss (MSE)
├─ Smoothness Loss (velocity regularization)
└─ Acceleration Loss (smooth motion)
```

**Key Design Decisions:**

1. **Include Previous Robot State:** Input contains both skeleton history AND robot's previous joint configurations to ensure physically feasible trajectories

2. **Action Chunking:** Predicts 50-frame sequences rather than single timesteps, creating perceived intentionality and commitment to movements

3. **Predict Joint Angles (Not End-Effector):** Full 6-DOF configuration preserves the kinesthetic quality essential for physical theatre mirroring

4. **Overlap Execution:** Re-predict every 10 frames with 40-frame overlap for smooth transitions between chunks

---

## Four-Step Implementation Plan

### STEP 1: Docker Environment Setup

**Objective:** Create containerized development environment with ROS2, PyTorch CUDA, and JupyterLab that communicates with existing ZED2 container.

**Key Components:**
- Ubuntu 22.04 base with CUDA 12.1
- ROS2 Humble (desktop + rosbag tools)
- PyTorch with CUDA support
- JupyterLab (port 8888)
- URDF libraries (yourdfpy or roboticstoolbox-python)
- Visualization tools (Plotly, Matplotlib)

**Deliverables:**
- Dockerfile with all dependencies
- docker-compose.yml for multi-container orchestration
- Shared volume configuration
- GPU verification and ROS2 communication tests

---

### STEP 2: Data Processing Pipeline

**Objective:** Convert rosbag recordings into synchronized CSV files suitable for model training.

**Workflow:**

**Phase 1: One-Time Setup** (`00_setup_urdf.ipynb`)
- Extract URDF from `/robot_description` topic → save to `config/ur10.urdf`
- Extract static transform from `/tf_static` (verify against manual calibration)
- Validate forward kinematics implementation
- Test visualization with zero configuration

**Phase 2: Rosbag Processing** (`01_process_rosbags.ipynb`)
1. Load rosbag and extract topics (skeletons, joint_states)
2. Synchronize messages by timestamp (tolerance: 50ms, interpolate joints to skeleton rate)
3. **Interactive skeleton selection:**
   - 3D visualization showing Skeleton 0 (red), Skeleton 1 (blue), UR10 robot
   - User selects which skeleton is the "leader"
   - Validate selection exists throughout recording
4. Transform selected skeleton to robot_base_link frame (using static calibration)
5. Quality checks (tracking confidence, position jumps, joint limits)
6. Export to CSV: `timestamp, sk_0_x...sk_15_z, j0...j5` (55 columns)
7. Update master metadata CSV with recording info

**Coordinate Transform:**
- Static transform provided manually in config: `camera → robot_base_link`
- Applied once during export using quaternion + translation
- All exported data in robot_base_link frame

**Visualization Module:**
- `ur10_kinematics.py`: Forward kinematics using extracted URDF
- `robot_viz.py`: 3D visualization of robot at given joint configuration
- `skeleton_viz.py`: Skeleton keypoint visualization
- `combined_viz.py`: Skeleton + robot in same coordinate frame

**Deliverables:**
- Processed CSV files (one per rosbag)
- Master metadata CSV (recordings index)
- Saved URDF file
- Visualization utilities
- Data quality reports

---

### STEP 3: Model Development & Training

**Objective:** Train Action Chunking Transformer to predict robot trajectories from skeleton sequences.

**Workflow:**

**Phase 1: Data Preparation** (`02_prepare_training_data.ipynb`)
1. Load all processed CSVs
2. Create temporal windows:
   - Input: 10 frames of [skeleton + robot state]
   - Output: 50 frames of future robot joints
   - Sliding window with stride=1 (dense sampling)
3. Normalize data:
   - Skeleton: Center at hip, scale to robot workspace
   - Joints: Already in radians [-π, π]
4. Train/val/test split **by recording session** (70/15/15)
   - Critical: Split by recording, not random, to test generalization
5. Create PyTorch Dataset with augmentation:
   - Temporal jitter (±2 frames)
   - Skeleton noise (Gaussian σ=0.01m)
6. Save preprocessed tensors

**Phase 2: Model Training** (`03_train_vam.ipynb`)
1. Implement Action Chunking Transformer architecture:
   - Skeleton encoder (temporal transformer)
   - Robot encoder (MLP)
   - Fusion layer
   - Action decoder (transformer)
2. Training loop:
   - Optimizer: AdamW, lr=1e-4
   - Scheduler: CosineAnnealing
   - Loss: Prediction + 0.1×Smoothness + 0.05×Acceleration
   - Batch size: 32
   - Epochs: 200
3. Logging with TensorBoard
4. Checkpoint best validation loss

**Phase 3: Evaluation**
1. Quantitative metrics:
   - Per-joint MSE
   - End-effector position error
   - Trajectory smoothness score
   - Fréchet distance
2. Qualitative visualization:
   - Predicted vs ground truth robot overlaid
   - Side-by-side comparisons with skeleton
   - Export videos of predictions
3. Attention analysis:
   - Which skeleton keypoints does model attend to?
   - Visualize attention heatmaps
4. Test on held-out performer (if available)

**Inference Function:**
```python
predict_chunk(skeleton_history, robot_history) 
  → predicted_trajectory [50, 6]
```

**Deliverables:**
- Trained model checkpoint (`data/models/vam_best.pth`)
- Training notebooks with full pipeline
- Evaluation results and visualizations
- Model architecture module (`vam_model/`)

---

### STEP 4: Real-Time Inference Pipeline

**Objective:** Deploy trained model as ROS2 node for live performance.

**ROS2 Node Architecture:**

**Main Node** (`vam_inference_node`)
- **Subscribes:**
  - `/zed/zed_node/body_trk/skeletons` (skeleton tracking)
  - `/joint_states` (current robot state)
- **Publishes:**
  - `/vam/predicted_trajectory` (JointTrajectory message)
  - `/vam/visualization_markers` (RViz markers)
- **Parameters:**
  - Model checkpoint path
  - Selected skeleton ID
  - Inference rate (30 Hz)
  - Safety thresholds

**Processing Pipeline:**
1. Maintain rolling buffer (T_in=10 frames) of skeleton + robot state
2. When buffer full → run inference
3. Predict 50-frame trajectory chunk
4. Apply smoothing (Savitzky-Golay filter)
5. Blend with previous prediction (overlap=40 frames)
6. Safety checks (joint limits, velocities, accelerations)
7. Publish trajectory to robot controller
8. Publish visualization markers

**Supporting Modules:**
- `model_wrapper.py`: Load PyTorch model, handle inference
- `transform_utils.py`: Coordinate transforms, ROS message creation
- `trajectory_smoother.py`: Savitzky-Golay filter, chunk blending
- `safety_checker.py`: Validate trajectories against limits

**Execution Modes:**

**Mode 1: Rosbag Testing**
```bash
ros2 bag play test_recording.db3
ros2 launch vam_inference vam_inference.launch.py
```
- Validate predictions match training evaluation
- Measure inference latency (target: <50ms)
- Visualize in RViz

**Mode 2: Live Deployment**
```bash
ros2 launch vam_inference vam_inference.launch.py
```
- Connect to live ZED2 tracking
- Send commands to UR10
- Monitor safety continuously

**Safety Features:**
- Joint limit validation
- Velocity/acceleration constraints
- Emergency stop if tracking lost
- Workspace boundary checks

**Deliverables:**
- ROS2 package (`vam_inference/`)
- Launch files and configurations
- RViz visualization setup
- Testing and profiling scripts
- Deployment documentation

---

## Key Technical Decisions

### 1. Why Action Chunking Transformer?

**Selected Approach:** Transformer-based action chunking

**Rationale:**
- Predicts sequences (50 frames) rather than single timesteps
- Creates perceived intentionality through trajectory commitment
- Faster inference than diffusion models
- Better temporal coherence than frame-by-frame prediction
- Proven success in robotic manipulation tasks

**Alternative Considered:** Diffusion-based policies
- Rejected: More complex, higher latency, overkill for current dataset size

### 2. Input Representation

**Decision:** Include both skeleton history AND robot state history

**Rationale:**
- Robot's response should depend on its current configuration
- Prevents impossible jumps between poses
- Provides implicit velocity information
- Improves physical feasibility of predictions
- Better temporal coherence

### 3. Prediction Target

**Decision:** Predict all 6 joint angles (not just end-effector)

**Rationale:**
- Physical theatre requires full-body configuration matching
- Robot's "posture" is what creates theatrical mirroring
- Avoids IK ambiguities that could break perceived relationship
- Preserves the kinesthetic quality essential for complicité

### 4. URDF Source

**Decision:** Extract URDF from `/robot_description` topic in rosbag

**Rationale:**
- Exact model used by actual robot system
- Includes any custom end-effector configurations
- Guaranteed consistency with robot kinematics
- No version mismatches

### 5. Coordinate Frame

**Decision:** All data in robot_base_link frame

**Rationale:**
- Robot-centric workspace makes sense for robot control
- Static transform (camera→robot) applied once during processing
- Consistent frame for all training and inference
- Simplifies spatial reasoning

---

## Success Criteria

### Technical Metrics

**Quantitative:**
- Model test MSE: TBD (establish baseline first)
- Inference latency: <50ms (for 30Hz real-time operation)
- Trajectory smoothness: Velocity variation <TBD
- No safety violations during testing

**Performance:**
- Training time: ~1 hour per epoch on RTX 5070 Ti
- Data processing: >1× realtime (faster than recording speed)
- GPU memory: Model fits in VRAM with batch_size=32

### Artistic Evaluation (Primary)

**Most Critical Success Factors:**
1. **Perceived Agency:** Does the robot feel intentional rather than reactive?
2. **Complicité:** Can performers develop ensemble connection with the robot?
3. **Readability:** Are the robot's movements interpretable by human partners?
4. **Improvisation:** Does it enable genuine co-creation vs. scripted response?

**Evaluation Method:**
- Qualitative assessment by choreographers (Piumi & team)
- Workshop sessions with performers
- Documentation of successful improvised sequences
- Comparison with pre-programmed RAPP Lab 03 performances

### Documentation & Reproducibility

- All code documented and tested
- Setup instructions enable reproduction
- Performance characteristics documented
- Learnings captured for publication (ICSR 2026, journal paper)

---

## Risk Mitigation

### Technical Risks

**Risk 1: Model doesn't generalize to new performers**
- Mitigation: Diverse training data, test on held-out performers, augmentation
- Fallback: Collect more data, fine-tune for specific performers

**Risk 2: Inference too slow for real-time**
- Mitigation: Profile early, optimize model size, use efficient libraries
- Fallback: Reduce T_out (shorter predictions), model quantization, ONNX/TensorRT

**Risk 3: Robot movements feel random/jerky**
- Mitigation: Smoothness in loss function, action chunking architecture, trajectory blending
- Fallback: Increase smoothing weight, longer chunk overlap, additional filtering

**Risk 4: Safety concerns in live deployment**
- Mitigation: Extensive rosbag testing first, conservative velocity limits, emergency stop
- Fallback: Human-in-the-loop approval, reduced speed operation

### Artistic Risks

**Risk 1: Robot lacks perceived agency**
- Mitigation: Action chunking creates commitment, CVAE for variation (future)
- Fallback: Adjust prediction horizon, incorporate stochasticity, hybrid scripted/learned

**Risk 2: Insufficient training data quality**
- Mitigation: Data quality checks, collection guidelines, iterative improvement
- Fallback: Collect additional high-quality recordings, focused on failure modes

---

## Project Timeline

### Phase 1: Infrastructure (Weeks 1-2)
- Docker environment setup
- URDF extraction and validation
- Visualization module development
- Test with sample rosbag

### Phase 2: Data Pipeline (Weeks 3-4)
- Process all existing rosbags
- Quality validation and statistics
- Create PyTorch training datasets
- Verify data characteristics

### Phase 3: Model Development (Weeks 5-6)
- Implement ACT architecture
- Train initial model
- Evaluate performance
- Iterate on hyperparameters

### Phase 4: Deployment Preparation (Week 7)
- Develop ROS2 inference node
- Test with rosbag playback
- Safety validation
- Performance profiling

### Phase 5: Live Testing (Week 8+)
- Gradual live deployment
- Workshop with performers
- Collect feedback and failure cases
- Iterate on model/data

---

## Deliverables Summary

### Software Artifacts

**Docker Environment:**
- Dockerfile with all dependencies
- docker-compose.yml for orchestration
- Setup and testing documentation

**Data Processing:**
- Jupyter notebooks (00, 01)
- Visualization module (`visualization/`)
- Processed CSV datasets
- Master metadata file

**Model Development:**
- Jupyter notebooks (02, 03)
- Model architecture module (`vam_model/`)
- Trained model checkpoint
- Evaluation results and visualizations

**Inference System:**
- ROS2 package (`vam_inference/`)
- Launch files and configurations
- RViz visualization
- Testing and profiling tools

### Documentation

**Technical:**
- Setup and installation guides
- API documentation for modules
- Coordinate frame definitions
- UR10 specifications reference

**Research:**
- Data collection guidelines
- Training procedures and results
- Performance benchmarks
- Artistic evaluation methodology

**Deployment:**
- Safety protocols
- Troubleshooting guide
- Parameter tuning documentation
- Live deployment procedures

---

## Future Directions

### Immediate Next Steps (Post-RAPP Lab 04)

**Enhanced Agency:**
- Integrate CVAE for controlled stochasticity
- Explore limited autonomy in movement initiation
- Develop anticipation mechanisms

**Expanded Training:**
- Collect data with more performers
- Include contrasting movements (not just mirroring)
- Different attitudes and dynamic qualities
- South Asian sociocultural narratives

**Technical Improvements:**
- Model compression for faster inference
- Attention mechanism refinement
- Multi-modal inputs (sound, proximity)

### Long-Term Vision

**RAPP Lab 05 and Beyond:**
- Multiple robot ensemble coordination
- Audience interaction and response
- Cultural narrative diversity
- Public performance demonstrations
- Integration with NIDA Future Centre workshops

---

## Appendices

### A. Configuration Templates

**data_processing_config.yaml:**
```yaml
# Static transform (USER FILLS IN CALIBRATION)
camera_to_robot_transform:
  translation: [x, y, z]  # meters
  rotation: [qx, qy, qz, qw]  # quaternion

# Topics
topics:
  skeleton: "/zed/zed_node/body_trk/skeletons"
  joint_states: "/joint_states"
  robot_description: "/robot_description"

# Processing parameters
sync_tolerance_sec: 0.05
skeleton_confidence_threshold: 0.5
```

**training_config.yaml:**
```yaml
# Model architecture
model:
  T_in: 10
  T_out: 50
  skeleton_encoder:
    n_layers: 6
    n_heads: 8
    d_model: 256
  robot_encoder:
    hidden_dims: [64, 128]
  decoder:
    n_layers: 4
    n_heads: 8

# Training
training:
  batch_size: 32
  learning_rate: 1e-4
  epochs: 200
  loss_weights:
    prediction: 1.0
    smoothness: 0.1
    acceleration: 0.05
```

### B. Skeleton Keypoint Definition

**ZED 2i Body Tracking - 16 Keypoints:**
```
0:  PELVIS (hip center)
1:  NAVAL_SPINE
2:  CHEST_SPINE
3:  NECK
4:  LEFT_CLAVICLE
5:  LEFT_SHOULDER
6:  LEFT_ELBOW
7:  LEFT_WRIST
8:  RIGHT_CLAVICLE
9:  RIGHT_SHOULDER
10: RIGHT_ELBOW
11: RIGHT_WRIST
12: LEFT_HIP
13: LEFT_KNEE
14: RIGHT_HIP
15: RIGHT_KNEE

Note: Wrist and knee are terminal points in 16-keypoint mode
```

### C. UR10 Joint Naming

**Joint Order (0-indexed):**
```
j0: Base (shoulder pan)
j1: Shoulder lift
j2: Elbow
j3: Wrist 1
j4: Wrist 2
j5: Wrist 3
```

### D. File Naming Conventions

**Rosbags:**
```
YYYYMMDD_performerID_attitude_takeNumber.db3
Example: 20250115_performerA_diagonal_001.db3
```

**Processed CSVs:**
```
rec_XXX.csv (where XXX is zero-padded recording ID)
Example: rec_001.csv
```

**Model Checkpoints:**
```
vam_YYYYMMDD_HHMM.pth (timestamp)
vam_best.pth (best validation loss)
```

---

## Contact & Support

**Project Lead:** Dr. Maleen Jayasuriya  
**Institution:** University of Canberra - Collaborative Robotics Lab  
**Context:** RAPP Lab 04 (2025-2026)

**Related Publications:**
- RAPP Lab 03: "Exploring Dramaturgical Potential in Human-Robot Ensembles" (ICRA 2025)
- RAPP Lab Retrospective: Cultural Robotics volume (in press)

**Upcoming Presentations:**
- ICSR 2026 Workshop (July, London)
- NIDA Future Centre Workshops (March 2026, Sydney)

---

*This design brief represents the starting point for RAPP Lab 04 development. Details may evolve based on experimental findings and artistic requirements.*

**Document Version:** 1.0  
**Last Updated:** December 2025  
**Status:** Initial Planning Phase
