from jetracer.nvidia_racecar import NvidiaRacecar

car = NvidiaRacecar()

#%%

car.steering = 0


#%%

print(car.steering_gain)

#%%

print(car.steering_offset)

#%%
car.throttle = 0
car.throttle = 1


#%%
car.throttle = 0
#%%

print(car.throttle_gain)

#%%

car.throttle_gain = 0.3

#%%
car.throttle_gain = 0