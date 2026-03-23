module CollisionUtil
open System
open System.Numerics

let private eps = 1e-8f

/// Rotate the body-forward (1,0,0) by quaternion q. Assumes body-forward is +X.
let quaternionToForward (q: Quaternion) : Vector3 =
    // Expand terms for numerical stability
    let xx = q.X * q.X
    let yy = q.Y * q.Y
    let zz = q.Z * q.Z
    let xy = q.X * q.Y
    let xz = q.X * q.Z
    let yz = q.Y * q.Z
    let wx = q.W * q.X
    let wy = q.W * q.Y
    let wz = q.W * q.Z

    let fx = 1.0f - 2.0f * (yy + zz)
    let fy = 2.0f * (xy + wz)
    let fz = 2.0f * (xz - wy)

    let forward = Vector3(fx, fy, fz)
    if forward = Vector3.Zero then Vector3.UnitX
    else Vector3.Normalize forward

/// Return true if the impact point lies in front of the vehicle.
/// threshold is cosine of allowed cone (e.g., 0.5 ~ 60 degrees).
let isImpactInFront (carPos: Vector3) (carOrient: Quaternion) (impactPoint: Vector3 option) (threshold: float32) : bool =
    match impactPoint with
    | None -> false
    | Some impact ->
        let dir = Vector3.Subtract(impact, carPos)
        if dir.LengthSquared() < eps then false
        else
            let dirN = Vector3.Normalize dir
            let forward = quaternionToForward carOrient
            let dot = Vector3.Dot(forward, dirN)
            dot > threshold

/// Fallback: use collision normal and vehicle velocity to infer a frontal collision.
/// normal: surface normal pointing away from surface.
/// normalThreshold: e.g., 0.6; velocityThreshold: minimum speed to consider (m/s).
let isCollisionFacingFrontByNormal (carVelocity: Vector3) (carOrient: Quaternion) (normal: Vector3) (normalThreshold: float32) (velocityThreshold: float32) : bool =
    if normal.LengthSquared() < eps then false
    elif carVelocity.LengthSquared() < velocityThreshold * velocityThreshold then false
    else
        let forward = quaternionToForward carOrient
        let normalN = Vector3.Normalize normal
        // surface faces the vehicle front if forward · (-normal) is large
        let dotFacing = Vector3.Dot(forward, Vector3.Negate normalN)
        let velDir = Vector3.Normalize carVelocity
        let velAlongForward = Vector3.Dot(velDir, forward)
        let ret = dotFacing > normalThreshold && velAlongForward > 0.0f
        ret

/// Reflect an incoming velocity about the collision normal (perfect elastic reflection).
let reflect (incoming: Vector3) (normal: Vector3) : Vector3 =
    if normal.LengthSquared() < eps then incoming
    else
        let n = Vector3.Normalize normal
        Vector3.Subtract(incoming, Vector3.Multiply(2.0f * Vector3.Dot(incoming, n), n))
