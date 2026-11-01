"""Configuration knobs: QUICK (default, seconds) and FULL (minutes)."""

QUICK = {
    "teacher_steps": 2500, "teacher_odes": 50, "student_steps": 2000, "batch": 256, "width": 128,
}

FULL = {
    "teacher_steps": 10000, "teacher_odes": 50, "student_steps": 8000, "batch": 256, "width": 256,
}
