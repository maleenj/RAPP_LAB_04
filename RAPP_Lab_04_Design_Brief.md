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
- Training GPU: NVIDIA RTX 5090 (Blackwell, sm_120)
- Inference GPU: NVIDIA RTX 5070 laptop (Blackwell, sm_120)
- OS: Ubuntu 22.04
- Containerization: Docker with GPU passthrough

**ROS2 Environment:**
- ROS2 Humble
- Network: Host mode for inter-container communication
- Domain ID: 0 (matching ZED2 container)

**Machine Learning:**
- Framework: PyTorch 2.10.0+cu128, CUDA 12.8
- Docker base: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`
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
- Frequency: Skeleton ~15Hz (after processing), Joints ~125Hz (raw)
- Coordinate Frame: Both in robot_base_link after transformation

**ROS2 Topics:**
```
/zed/zed_node/body_trk/skeletons  → Skeleton data (may contain 2 people)
/joint_states                      → UR10 joint positions
/robot_description                 → URDF (extract once)
/tf_static                         → Static transforms
```

**Data Collection Pattern:**
- Grid-based collection: 10 episodes across spatial positions and speeds
- Naming convention: `25_12_11_RAPP_M_R{radius}G{position}S{speed}_{take}.csv`
  - **R** = radius (metres from robot base)
  - **G** = arc position (1-5 on semicircular arc in front of robot)
  - **S** = speed level
- ~1200 frames per episode at ~15Hz (~80s each)
- Performer's absolute position relative to robot is meaningful — the model should respond differently based on spatial position (R/G values)

**Training Data Format:**
- Temporal sequences: Input window (T_in=10 frames) → Output prediction (T_out=10 frames)
- Input: [skeleton_history: [10, 48], robot_history: [10, 6]]
- Output: [robot_future: [10, 6]]
- At ~15Hz: T_in ≈ 0.67s history, T_out ≈ 0.67s prediction horizon

### Model Architecture

**Action Chunking Transformer:**

> **Note:** The architecture parameters below are starting points. With ~8,000 training windows (10 episodes), the model may be over-parameterized. Smaller configurations (2-3 layers, 4 heads, d_model=128) should be explored first during training to avoid overfitting. These will be tuned in notebook 03.

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
├─ Predicts entire trajectory chunk (10 frames)
└─ Output: [T_out=10, 6] joint angle predictions

Loss Function:
├─ Prediction Loss (MSE)
├─ Smoothness Loss (velocity regularization)
└─ Acceleration Loss (smooth motion)
```

**Key Design Decisions:**

1. **Include Previous Robot State:** Input contains both skeleton history AND robot's previous joint configurations to ensure physically feasible trajectories

2. **Action Chunking:** Predicts 10-frame sequences (~0.67s at 15Hz) rather than single timesteps, creating trajectory commitment while maintaining tight responsive control through frequent re-prediction

3. **Predict Joint Angles (Not End-Effector):** Full 6-DOF configuration preserves the kinesthetic quality essential for physical theatre mirroring

4. **Overlap Execution:** Re-predict every 3-5 frames (~0.2-0.33s) with overlap for smooth transitions between chunks

---

## Four-Step Implementation Plan

### STEP 1: Docker Environment Setup

**Objective:** Create containerized development environment with ROS2, PyTorch CUDA, and JupyterLab that communicates with existing ZED2 container.

**Key Components:**
- Ubuntu 22.04 base with CUDA 12.8
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
6. Export to CSV: `timestamp, sk_0_x...sk_15_z, j0...j5, ee_x...ee_yaw` (61 columns)
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
1. Load all processed CSVs (10 episodes, ~1200 frames each)
2. Create temporal windows (episode-aware, never cross episode boundaries):
   - Input: 10 frames of [skeleton + robot state] = [T_in, 54]
   - Output: 10 frames of future robot joints = [T_out, 6]
   - Sliding window with stride=1 (dense sampling)
3. Normalize data:
   - Standardize only (zero mean, unit variance from training episodes)
   - No hip-centering — absolute position in robot_base_link frame preserves meaningful spatial relationships from grid data collection pattern
   - Both skeleton and joints standardized the same way
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
  → predicted_trajectory [10, 6]
```

**Deliverables:**
- Trained model checkpoint (`data/models/vam_best.pth`)
- Training notebooks with full pipeline
- Evaluation results and visualizations
- Model architecture module (`vam_model/`)

---

### STEP 4: Real-Time Inference Pipeline

**Objective:** Deploy trained model for live performance with smooth, continuous robot motion.

#### Smooth Control Strategy: Temporal Ensemble with Receding Horizon

The model predicts 10-frame chunks (~0.67s). Executing full chunks sequentially produces jerky motion at boundaries. The solution is **temporal ensemble** (from ACT paper, Tony Zhao et al.):

1. Re-predict every K=2 frames (~133ms), not every 10 frames
2. Each prediction still produces a full 10-frame chunk
3. Multiple overlapping chunks cover each timestep (up to 5 with K=2)
4. Blend all overlapping predictions using exponential decay weighting

```
Time:     t0  t1  t2  t3  t4  t5  t6  t7  t8  t9  t10 t11
Chunk A:  [0   1   2   3   4   5   6   7   8   9]
Chunk B:            [0   1   2   3   4   5   6   7   8   9]
Chunk C:                    [0   1   2   3   4   5   6   7   8   9]

At t4: blend A[4], B[2], C[0] → weights exp(-λ*4), exp(-λ*2), exp(-λ*0)
```

**Why this works:** Averaging overlapping predictions filters noise and eliminates chunk boundary discontinuities. No post-hoc filtering needed (no Savitzky-Golay, no phase lag). The velocity/acceleration losses in training ensure within-chunk smoothness; the temporal ensemble ensures between-chunk continuity.

**Tunable parameters:**
| Parameter | Default | Effect |
|-----------|---------|--------|
| K (stride) | 2 | Lower = responsive, higher = stable |
| λ (decay) | 0.01 | Lower = smoother, higher = responsive |
| Warmup | 20 frames | Buffer fill + ensemble history before execution |

#### Execution Interface: FollowJointTrajectory

**NOT MoveIt MoveGroup** — waypoint-by-waypoint MoveGroup (plan → execute → stop → repeat) produces start-stop jerkiness. Instead:

- **`FollowJointTrajectory` action** for trajectory execution — the standard ROS2 interface used by UR, TM, Franka, and all arms. MoveIt uses this under the hood.
- **Future obstacle avoidance**: Add MoveIt PlanningScene + OctoMap as a validation layer (check trajectory for collisions before sending to controller). This doesn't require using MoveGroup for execution.

```
Temporal Ensemble → blended targets → [future: collision check] → FollowJointTrajectory
```

#### Implementation Architecture

**Phase A: Core Inference Utilities** (`vam_utils/inference/`, no ROS2 dependency)
- `inference_config.py`: InferenceConfig dataclass (paths, K, λ, safety limits)
- `model_wrapper.py`: Load model + predict, denormalize to radians
- `temporal_ensemble.py`: Prediction buffer + exponential decay blending
- `input_assembler.py`: Rolling buffer + normalization for skeleton/joints
- `safety_checker.py`: Joint limits, velocity/acceleration constraints

**Phase B: Offline Testing** (`notebooks/04_inference_test.ipynb`)
- Replay test episodes through full pipeline, no ROS2
- Visualize ensemble vs raw predictions vs ground truth
- Parameter sweep (K, λ), latency profiling

**Phase C: ROS2 Inference Node** (`ros2_ws/src/vam_inference/`)

**Main Node** (`vam_inference_node`)
- **Subscribes:**
  - `/zed/zed_node/body_trk/skeletons` (skeleton tracking)
  - `/joint_states` (current robot state)
- **Publishes:**
  - `/vam/predicted_trajectory` (JointTrajectory message)
  - `/vam/joint_states` (for RViz visualization)
  - `/vam/visualization_markers` (RViz markers)

**Main loop (15Hz timer callback):**
1. Read latest skeleton + joint_states from buffers
2. Feed to InputAssembler (normalize + concat)
3. If buffer ready AND should_predict(): run inference, add to ensemble
4. Query TemporalEnsemble for blended target at current step
5. SafetyChecker: clamp limits, constrain velocity/acceleration
6. Publish (JointState for RViz mode, FollowJointTrajectory for robot mode)

**Supporting Modules (ROS2-specific):**
- `skeleton_buffer.py`: Subscribe to ZED skeleton topic, extract 48 floats
- `robot_state_buffer.py`: Subscribe to joint_states, handle joint name→index mapping
- `trajectory_publisher.py`: RViz mode (publish JointState) / Robot mode (FollowJointTrajectory action)

#### Execution Modes

**Mode 1: Offline Replay** (notebook, no ROS2)
- Feed CSV data frame-by-frame through pipeline
- Validate smoothness and accuracy

**Mode 2: Rosbag Testing** (ROS2, RViz only)
```bash
ros2 bag play test_recording.db3 --clock
ros2 launch vam_inference vam_inference.launch.py mode:=rviz use_sim_time:=true
```

**Mode 3: Live + RViz** (ROS2, no physical robot)
```bash
ros2 launch vam_inference vam_inference.launch.py mode:=rviz
```

**Mode 4: Physical Robot** (start at conservative velocity)
```bash
ros2 launch vam_inference vam_inference.launch.py mode:=robot max_velocity:=0.3
```

#### Safety Architecture (Defense in Depth)

1. **Model output clamping**: `inverse_normalize_joints(clamp_to_limits=True)` — UR10 joint limits
2. **Velocity limiting**: Scale delta to respect max_joint_velocity_rad_s (default: 1.0)
3. **Acceleration limiting**: Constrain rate of velocity change
4. **Tracking loss detection**: If no skeleton for >0.5s → graceful deceleration to zero
5. **UR10 built-in safety**: Controller rejects trajectories violating internal limits

#### Deliverables
- Inference utilities (`vam_utils/inference/`)
- Offline testing notebook (`04_inference_test.ipynb`)
- ROS2 package (`vam_inference/`)
- Launch files, RViz config, parameter YAML
- Testing and profiling documentation

---

## Key Technical Decisions

### 1. Why Action Chunking Transformer?

**Selected Approach:** Transformer-based action chunking

**Rationale:**
- Predicts sequences (10 frames / ~0.67s) rather than single timesteps
- Creates trajectory commitment while enabling frequent re-prediction for responsiveness
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
- Training time: ~1 hour per epoch on RTX 5090
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
- Fallback: Model quantization, ONNX/TensorRT (T_out already minimized to 10)

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
# Model architecture (starting point — tune during training)
model:
  T_in: 10
  T_out: 10
  skeleton_encoder:
    n_layers: 6     # Note: may need reducing for ~8K training samples
    n_heads: 8
    d_model: 256    # Consider d_model=128 with fewer layers first
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
YY_MM_DD_RAPP_M_R{radius}G{position}S{speed}_{take}.db3
Example: 25_12_11_RAPP_M_R2G1S1_001.db3
```

**Processed CSVs:**
```
YY_MM_DD_RAPP_M_R{radius}G{position}S{speed}_{take}.csv
Example: 25_12_11_RAPP_M_R2G1S1_001.csv
  R = radius in metres from robot base
  G = arc position (1-5 on semicircular arc)
  S = speed level
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

**Document Version:** 2.0
**Last Updated:** February 2026
**Status:** Model Training Complete, Real-Time Inference Next
