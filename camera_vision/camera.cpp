



struct Point3D {
    float x; 
    float y; 
    float z;
};

struct BiometricFrame {
    double timestamp;           // Tid (s)
    double elbow_flexion;       // Armbågens vinkel (grader)
    double shoulder_flexion;    // Axelns vinkel (grader)
    double wrist_velocity;      // Hastighet (mm/s)
    double wrist_acceleration;  // Acceleration (mm/s^2)
    double wrist_jerk;          // Ryckighet (mm/s^3)
};
