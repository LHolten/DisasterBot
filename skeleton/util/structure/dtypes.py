import numpy as np
from rlbot.utils.structures.game_data_struct import (
    MAX_NAME_LENGTH,
    ScoreInfo,
    BoxShape,
    CollisionShape,
    DropShotInfo,
    BoostPadState,
    TileInfo,
    TeamInfo,
    GameInfo,
    MAX_PLAYERS,
    MAX_BOOSTS,
    MAX_TILES,
    MAX_TEAMS,
    MAX_GOALS,
)
from rlbot.utils.structures.ball_prediction_struct import MAX_SLICES

dtype_Vector3 = np.dtype(("<f4", 3))

# game tick packet
dtype_Physics = np.dtype(
    [
        ("location", dtype_Vector3),
        ("rotation", dtype_Vector3),
        ("velocity", dtype_Vector3),
        ("angular_velocity", dtype_Vector3),
    ]
)

dtype_Name = np.dtype(("S2", (MAX_NAME_LENGTH,)))

dtype_Touch = np.dtype(
    [
        ("player_name", dtype_Name),
        ("time_seconds", "<f4"),
        ("hit_location", dtype_Vector3),
        ("hit_normal", dtype_Vector3),
        ("team", "<i4"),
        ("player_index", "<i4"),
    ]
)


dtype_ScoreInfo = np.dtype(ScoreInfo)

dtype_BoxShape = np.dtype(BoxShape)
dtype_CollisionShape = np.dtype(CollisionShape)


dtype_PlayerInfo = np.dtype(
    {
        "names": [
            "physics",
            "score_info",
            "is_demolished",
            "has_wheel_contact",
            "is_super_sonic",
            "is_bot",
            "jumped",
            "double_jumped",
            "name",
            "team",
            "boost",
            "hitbox",
            "hitbox_offset",
            "spawn_id",
        ],
        "formats": [
            dtype_Physics,
            dtype_ScoreInfo,
            "?",
            "?",
            "?",
            "?",
            "?",
            "?",
            dtype_Name,
            "u1",  # This part is the reason for the dictionary and the offsets.
            "<i4",
            dtype_BoxShape,
            dtype_Vector3,
            "<i4",
        ],
        "offsets": [0, 48, 76, 77, 78, 79, 80, 81, 82, 146, 148, 152, 164, 176],
    }
)

dtype_DropShotInfo = np.dtype(DropShotInfo)

dtype_BallInfo = np.dtype(
    [
        ("physics", dtype_Physics),
        ("latest_touch", dtype_Touch),
        ("drop_shot_info", dtype_DropShotInfo),
        ("collision_shape", dtype_CollisionShape),
    ]
)

dtype_BoostPadState = np.dtype(BoostPadState)
dtype_TileInfo = np.dtype(TileInfo)
dtype_TeamInfo = np.dtype(TeamInfo)
dtype_GameInfo = np.dtype(GameInfo)


dtype_GameTickPacket = np.dtype(
    [
        ("game_cars", dtype_PlayerInfo * MAX_PLAYERS),
        ("num_cars", "<i4"),
        ("game_boosts", dtype_BoostPadState * MAX_BOOSTS),
        ("num_boost", "<i4"),
        ("game_ball", dtype_BallInfo),
        ("game_info", dtype_GameInfo),
        ("dropshot_tiles", dtype_TileInfo * MAX_TILES),
        ("num_tiles", "<i4"),
        ("teams", dtype_TeamInfo * MAX_TEAMS),
        ("num_teams", "<i4"),
    ]
)


# field info
dtype_BoostPad = np.dtype({"names": ["location", "is_full_boost"], "formats": [dtype_Vector3, "?"], "itemsize": 16})


dtype_GoalInfo = np.dtype(
    {
        "names": ["team_num", "location", "direction", "width", "height"],
        "formats": ["u1", dtype_Vector3, dtype_Vector3, "<f4", "<f4"],  # same thing here
        "offsets": [0, 4, 16, 28, 32],
    }
)

dtype_FieldInfoPacket = np.dtype(
    [
        ("boost_pads", dtype_BoostPad * MAX_BOOSTS),
        ("num_boosts", "<i4"),
        ("goals", dtype_GoalInfo * MAX_GOALS),
        ("num_goals", "<i4"),
    ]
)

# ball prediction
dtype_Slice = np.dtype([("physics", dtype_Physics), ("game_seconds", "<f4")])

dtype_BallPrediction = np.dtype([("slices", dtype_Slice * MAX_SLICES), ("num_slices", "<i4")])


full_boost_dtype = np.dtype(
    {
        "names": ["location", "is_full_boost", "is_active", "timer"],
        "formats": [dtype_Vector3, "?", "?", "<f4"],
        "itemsize": 24,
        "aligned": True,
    }
)
