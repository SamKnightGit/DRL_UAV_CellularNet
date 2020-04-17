from mobile_env_original import MobiEnvironment
import matplotlib.pyplot as plt
num_base_stations = 4
num_users = 40
arena_width = 100


test_env = MobiEnvironment(num_base_stations, num_users, arena_width, "read_trace", "./ue_trace_10k.npy")
train_env = MobiEnvironment(num_base_stations, num_users, arena_width)

print("Training Environment")
print(f"Base Station Loc: {train_env.bsLoc}")
print(f"User Location: {train_env.ueLoc}")
plt.figure()
plt.xlim(0, 100)
plt.ylim(0, 100)
for point in train_env.bsLoc:
    plt.plot(point[0], point[1])
plt.show()

print("Testing Environment")
print(f"Base Station Loc: {test_env.bsLoc}")
print(f"User Location: {test_env.ueLoc}")