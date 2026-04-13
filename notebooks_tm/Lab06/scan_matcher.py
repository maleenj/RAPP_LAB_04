"""
Correlation-Based Scan Matching (Week 7) — INSTRUCTOR SOLUTION

Complete implementation with all TODO sections filled in.
For testing the lab pipeline only — do NOT distribute to students.

References:
  - Olson, "Real-Time Correlative Scan Matching" (2009)
"""

import numpy as np
from typing import Tuple
from scipy.ndimage import maximum_filter


class ScanMatcher:
    """
    Correlation-based scan matcher.

    Aligns two laser scans by searching over candidate relative poses
    and scoring each one using grid correlation.
    """

    def __init__(self,
                 search_x: float,
                 search_y: float,
                 search_theta: float,
                 resolution_x: float,
                 resolution_y: float,
                 resolution_theta: float,
                 local_grid_size: int,
                 local_grid_resolution: float,
                 min_score: float):
        self.search_x = search_x
        self.search_y = search_y
        self.search_theta = search_theta
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.resolution_theta = resolution_theta
        self.local_grid_size = local_grid_size
        self.local_grid_resolution = local_grid_resolution
        self.min_score = min_score

    def _build_local_grid(self, scan_points: np.ndarray) -> np.ndarray:
        """
        Rasterise scan points into a local occupancy grid for fast correlation.

        Args:
            scan_points: Nx2 array of (x, y) points in local frame

        Returns:
            2D float32 grid (1.0 where scan points land, 0.0 elsewhere)
        """
        size = self.local_grid_size
        res = self.local_grid_resolution
        offset = size // 2

        grid = np.zeros((size, size), dtype=np.float32)

        if len(scan_points) == 0:
            return grid

        cols = (scan_points[:, 0] / res).astype(int) + offset
        rows = (scan_points[:, 1] / res).astype(int) + offset

        mask = (rows >= 0) & (rows < size) & (cols >= 0) & (cols < size)
        grid[rows[mask], cols[mask]] = 1.0

        # Dilate by 1 cell for tolerance against discretisation errors
        grid = maximum_filter(grid, size=3)

        return grid

    def _score_alignment(self, grid: np.ndarray, scan_points: np.ndarray,
                         pose: np.ndarray) -> float:
        """
        Score how well scan_points align with the reference grid
        when transformed by the candidate pose.

        Args:
            grid:        Reference scan's local occupancy grid
            scan_points: Nx2 array of new scan points (local frame)
            pose:        Candidate relative pose [dx, dy, dtheta]

        Returns:
            Correlation score (count of overlapping points)
        """
        dx, dy, dtheta = pose
        size = self.local_grid_size
        res = self.local_grid_resolution
        offset = size // 2

        cos_dt = np.cos(dtheta)
        sin_dt = np.sin(dtheta)

        px = scan_points[:, 0]
        py = scan_points[:, 1]
        px_new = cos_dt * px - sin_dt * py + dx
        py_new = sin_dt * px + cos_dt * py + dy

        cols = (px_new / res).astype(int) + offset
        rows = (py_new / res).astype(int) + offset

        mask = (rows >= 0) & (rows < size) & (cols >= 0) & (cols < size)
        return float(np.sum(grid[rows[mask], cols[mask]]))

    def match(self, scan_ref: np.ndarray, scan_new: np.ndarray,
              initial_guess: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Match two scans using correlation-based grid search.

        Args:
            scan_ref:      Nx2 reference scan points in local frame
            scan_new:      Mx2 new scan points in local frame
            initial_guess: Initial relative pose estimate [dx, dy, dtheta]

        Returns:
            best_pose:  Best relative pose [dx, dy, dtheta]
            covariance: 3x3 covariance matrix of the match
            score:      Normalised score in [0, 1] (0 if match rejected)
        """
        default_cov = np.diag([0.1, 0.1, 0.05])

        if len(scan_ref) == 0 or len(scan_new) == 0:
            return initial_guess.copy(), default_cov, 0.0

        # Step 1: Build local occupancy grid from reference scan
        ref_grid = self._build_local_grid(scan_ref)

        # Step 2: Generate search grid around initial guess
        dx_guess, dy_guess, dtheta_guess = initial_guess

        x_values = np.arange(
            dx_guess - self.search_x,
            dx_guess + self.search_x + self.resolution_x * 0.5,
            self.resolution_x
        )
        y_values = np.arange(
            dy_guess - self.search_y,
            dy_guess + self.search_y + self.resolution_y * 0.5,
            self.resolution_y
        )
        theta_values = np.arange(
            dtheta_guess - self.search_theta,
            dtheta_guess + self.search_theta + self.resolution_theta * 0.5,
            self.resolution_theta
        )

        # Step 3: Exhaustive search over all candidate poses
        scores = {}
        best_score = -1.0
        best_pose = initial_guess.copy()
        best_idx = (0, 0, 0)

        for ix, x in enumerate(x_values):
            for iy, y in enumerate(y_values):
                for it, theta in enumerate(theta_values):
                    candidate = np.array([x, y, theta])
                    score = self._score_alignment(ref_grid, scan_new, candidate)

                    scores[(ix, iy, it)] = score

                    if score > best_score:
                        best_score = score
                        best_pose = candidate.copy()
                        best_idx = (ix, iy, it)

        # Step 4: Normalize score
        normalized_score = best_score / len(scan_new)

        # Step 5: Reject poor matches
        if normalized_score < self.min_score:
            return initial_guess.copy(), default_cov, 0.0

        # Step 6: Estimate covariance from the Hessian
        covariance = self._estimate_covariance_from_hessian(
            scores, best_idx,
            self.resolution_x, self.resolution_y, self.resolution_theta
        )

        return best_pose, covariance, normalized_score

    def _estimate_covariance_from_hessian(self,
                                          scores: dict,
                                          best_idx: Tuple[int, int, int],
                                          step_x: float,
                                          step_y: float,
                                          step_theta: float) -> np.ndarray:
        """
        Estimate match covariance from the Hessian of the score surface.

        Args:
            scores:     Dict mapping (ix, iy, it) tuples to score values
            best_idx:   (ix, iy, it) of the peak score
            step_x:     Search step in x (metres)
            step_y:     Search step in y (metres)
            step_theta: Search step in theta (radians)

        Returns:
            3x3 covariance matrix
        """
        default_cov = np.diag([0.1, 0.1, 0.05])

        f0 = scores.get(best_idx, 0.0)
        if f0 == 0.0:
            return default_cov

        steps = [step_x, step_y, step_theta]
        H = np.zeros((3, 3))

        # Diagonal elements: H[i,i] = (f+ - 2*f0 + f-) / step_i^2
        for i in range(3):
            idx_plus = list(best_idx)
            idx_minus = list(best_idx)
            idx_plus[i] += 1
            idx_minus[i] -= 1

            f_plus = scores.get(tuple(idx_plus), 0.0)
            f_minus = scores.get(tuple(idx_minus), 0.0)

            H[i, i] = (f_plus - 2.0 * f0 + f_minus) / (steps[i] ** 2)

        # Off-diagonal elements: H[i,j] = (f++ - f+- - f-+ + f--) / (4*si*sj)
        for i in range(3):
            for j in range(i + 1, 3):
                idx_pp = list(best_idx)
                idx_pm = list(best_idx)
                idx_mp = list(best_idx)
                idx_mm = list(best_idx)

                idx_pp[i] += 1; idx_pp[j] += 1
                idx_pm[i] += 1; idx_pm[j] -= 1
                idx_mp[i] -= 1; idx_mp[j] += 1
                idx_mm[i] -= 1; idx_mm[j] -= 1

                f_pp = scores.get(tuple(idx_pp), 0.0)
                f_pm = scores.get(tuple(idx_pm), 0.0)
                f_mp = scores.get(tuple(idx_mp), 0.0)
                f_mm = scores.get(tuple(idx_mm), 0.0)

                H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * steps[i] * steps[j])
                H[j, i] = H[i, j]

        # Covariance = (-H)^{-1}  (negate because H is concave at a maximum)
        neg_H = -H

        # Check positive definiteness
        eigenvalues = np.linalg.eigvalsh(neg_H)
        if np.any(eigenvalues <= 1e-6):
            return default_cov

        try:
            covariance = np.linalg.inv(neg_H)
        except np.linalg.LinAlgError:
            return default_cov

        # Sanity check
        cov_eigenvalues = np.linalg.eigvalsh(covariance)
        if np.any(cov_eigenvalues <= 0):
            return default_cov

        return covariance


# ============================================================================
# Helper function (provided — do not modify)
# ============================================================================

def scans_from_ranges(ranges: np.ndarray, angle_min: float,
                      angle_increment: float, min_range: float = 0.1,
                      max_range: float = 12.0,
                      lidar_yaw_offset: float = 0.0) -> np.ndarray:
    """
    Convert laser scan ranges to (x, y) points in the robot's local frame.
    """
    points = []
    for i, r in enumerate(ranges):
        if np.isnan(r) or r < min_range or r > max_range:
            continue
        beam_angle = lidar_yaw_offset + angle_min + i * angle_increment
        x = r * np.cos(beam_angle)
        y = r * np.sin(beam_angle)
        points.append([x, y])

    if len(points) == 0:
        return np.empty((0, 2))
    return np.array(points)
