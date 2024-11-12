import sutterMP285

sutter = sutterMP285.sutterMP285()
sutter.getStatus()
sutter.setVelocity(1000)
sutter.setOrigin()
posnew = [100, 100, 100]
sutter.gotoPosition(posnew)
sutter.updatePanel()