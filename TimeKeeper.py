import time
class timeKeeper:
    def __init__(self, name, totalLaps, incrementAmount = 1):
        self.startTime = time.time()
        self.name = name
        self.lapsDone = 0
        self.totalLaps = totalLaps
        self.incrementAmount = incrementAmount
        self.endTime = 0

    def timeElapsed(self):
        return time.time() - self.startTime
    
    def addLap(self, amount = 1):
        self.lapsDone += amount

    def getAvgLapTime(self): #Returns time in seconds
        avgTime = (time.time() - self.startTime) / (1 if self.lapsDone == 0 else self.lapsDone)  
        return avgTime

    def formattedTime(self, time):
        if time < 60: return (time, "Sec") #In seconds
        elif time < 3600: return (time / 60, "Min") #In minutes
        elif time < 60 * 60 * 24: return (time / 60 / 60, "Hours") #In hours
        else: return (time / 60 / 60 / 24, "Days") #In days

    def eta(self):
        avgTime = self.getAvgLapTime()
        lapsLeft = self.totalLaps - self.lapsDone
        timeLeft = avgTime * lapsLeft
        return self.formattedTime(timeLeft)

    def activeLap(self):
        self.addLap()
        return self.getAvgLapTime()
    
    def timerBroadcast(self):
        self.addLap(self.incrementAmount)
        time, context = self.eta()
        print(f"[{self.name}] | ETA {time:.2f} {context} | Progress: {((self.lapsDone / self.totalLaps)*100):.2f}%")

    def endTimer(self):
        self.endTime = time.time()
        lifetime = self.endTime - self.startTime
        formattedLifetime, context = self.formattedTime(lifetime)
        print(f"{self.name} finished in {formattedLifetime} {context}")
