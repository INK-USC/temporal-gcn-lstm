from collections import defaultdict
import sys
import timeit
import numpy as np

"""
Input format

'id,action,time'
...
 
ex. 'User1,send-snap,2018-06-28T01:00:00.000000000+0100'

Output format

'id # sessions
 action1,action2,probability
...
'

ex 'Send-Snap,Receive-Snap,0.5'

"""

SESSION_INPUT = sys.argv[1]
GRAPH_OUTPUT = sys.argv[2]

SECONDS_IN_DAY = 86400
SEVEN_DAYS = SECONDS_IN_DAY * 7

start_time = timeit.default_timer()


def compute_graph(gid, d, outfile, session_count):
    outfile.write(gid + ' ' + str(session_count) + ' sessions' + '\n')
    for action in d:
        total = sum(d[action].values())
        for k, v in d[action].items():
            outfile.write(','.join([action, k, str(v/total)]) + '\n')


with open(SESSION_INPUT, 'r') as infile, open(GRAPH_OUTPUT, 'w') as outfile:
    curr_gid = None
    prev_action = None
    d = defaultdict(lambda: defaultdict(lambda: 0))
    N = 0
    session_count = 0
    total_sessions = 0
    user_count = 0

    i_day = 0

    for line in infile:
        N += 1
        if N % 1000000 == 0:
            print(N, ' lines processed')
        gid, action, time = line.split(',')

        if gid != curr_gid:
            user_count += 1

            register_time = np.datetime64(time)

            if curr_gid is not None:
                for i in range(14 - i_day):
                    compute_graph(curr_gid, d, outfile, session_count)
                    d.clear()
                d.clear()
            prev_action = action
            curr_gid = gid
            total_sessions += session_count
            session_count = 1
            i_day = 0
        else:
            if action == 'SESSION_START' :
                session_count += 1
                if prev_action == 'SESSION_START':
                    d[prev_action]['SESSION_END'] += 1

                s_delta = (np.datetime64(time)-register_time).item().total_seconds() - i_day*SECONDS_IN_DAY
                days = int(s_delta/SECONDS_IN_DAY)

                if s_delta > SECONDS_IN_DAY:
                    for i in range(days):
                        compute_graph(curr_gid, d, outfile, session_count)
                        d.clear()
                    d.clear()
                    i_day += days

            else:
                d[prev_action][action] += 1

            prev_action = action

    compute_graph(curr_gid, d, outfile, session_count)

    outfile.write('Avg ' + str(total_sessions/user_count) + ' sessions for ' + str(user_count) + ' users' + '\n')

print('Time Elapsed: ', str(timeit.default_timer() - start_time))
