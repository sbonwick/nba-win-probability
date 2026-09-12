def parse_clock_to_seconds(clock:str) -> float:
    clock = clock.replace("PT","")
    minutes,seconds = clock.split("M")
    seconds = seconds.replace("S","")
    minutes = int(minutes)
    seconds = float(seconds)
    return minutes * 60 + seconds

def get_period_length(period:int) -> float:
    if period <= 4:
        return 12 * 60
    else:
        return 5 * 60

def get_gametime_elapsed(period:int, clock:str) -> float:
    period_length = get_period_length(period)
    time_remaining = parse_clock_to_seconds(clock)
    if period <= 4:
        return 48*60 - (period_length * (period - 1) + time_remaining)
    else:
        return 5 * 60 - time_remaining

def get_gametime_remaining(period:int, clock:str) -> float:
    period_length = get_period_length(period)
    time_remaining = parse_clock_to_seconds(clock)
    if period <= 4:
        return period_length * (4 - period) + time_remaining
    else:
        return time_remaining