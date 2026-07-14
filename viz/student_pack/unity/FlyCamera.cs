// FlyCamera.cs — simple free-fly camera to move around your 3D visuals.
//
// Controls:
//   W / A / S / D     move forward / left / back / right (along view direction)
//   Up / Down arrows  move straight up / down (world)
//   Hold Right Mouse  free-look (move the mouse to aim)
//   Left Shift        move faster (boost)
//
// SETUP: put this on your Camera (e.g. Main Camera). Press Play.
//
// NOTE: uses Unity's legacy Input. If you get an InputException, set
// Project Settings > Player > Active Input Handling to "Input Manager (Old)"
// or "Both". (Set `holdRightMouseToLook = false` for always-on mouse look.)

using UnityEngine;

public class FlyCamera : MonoBehaviour
{
    [Header("Movement")]
    public float moveSpeed = 3f;
    public float boostMultiplier = 3f;   // while holding Left Shift
    public float verticalSpeed = 3f;     // up/down arrows

    [Header("Look")]
    public float lookSensitivity = 2.5f;
    public bool holdRightMouseToLook = true;  // false = always look (cursor locked)
    public float pitchLimit = 89f;

    float _yaw;
    float _pitch;

    void Start()
    {
        Vector3 e = transform.eulerAngles;
        _yaw = e.y;
        _pitch = e.x;
        if (!holdRightMouseToLook) SetCursor(true);
    }

    void Update()
    {
        HandleLook();
        HandleMove();
    }

    void HandleLook()
    {
        bool looking = !holdRightMouseToLook || Input.GetMouseButton(1);

        if (holdRightMouseToLook)
        {
            if (Input.GetMouseButtonDown(1)) SetCursor(true);
            if (Input.GetMouseButtonUp(1)) SetCursor(false);
        }

        if (!looking) return;

        _yaw += Input.GetAxis("Mouse X") * lookSensitivity;
        _pitch -= Input.GetAxis("Mouse Y") * lookSensitivity;
        _pitch = Mathf.Clamp(_pitch, -pitchLimit, pitchLimit);
        transform.rotation = Quaternion.Euler(_pitch, _yaw, 0f);
    }

    void HandleMove()
    {
        float boost = Input.GetKey(KeyCode.LeftShift) ? boostMultiplier : 1f;

        // WASD along the camera's facing direction
        float h = Input.GetAxisRaw("Horizontal");   // A/D
        float v = Input.GetAxisRaw("Vertical");     // W/S
        Vector3 vel = (transform.forward * v + transform.right * h) * (moveSpeed * boost);

        // Up/Down arrows = world vertical
        float up = (Input.GetKey(KeyCode.UpArrow) ? 1f : 0f) - (Input.GetKey(KeyCode.DownArrow) ? 1f : 0f);
        vel += Vector3.up * (up * verticalSpeed * boost);

        transform.position += vel * Time.deltaTime;
    }

    void SetCursor(bool locked)
    {
        Cursor.lockState = locked ? CursorLockMode.Locked : CursorLockMode.None;
        Cursor.visible = !locked;
    }
}
