

if __name__ == '__main__':
    center = [0,0]
    width = 0.0005
    length = (0.01,0.01)
    up = ["Box_Ax", center[0] - length[0] / 2 - width, center[1] + length[1] / 2, center[0] + length[0] / 2 + width,
          center[1] + length[1] / 2 + width, 0.000000, -5.00e-7]
    down = ["Box_Ax", center[0] - length[0] / 2 - width, center[1] - length[1] / 2 - width, center[0] + length[0] / 2 + width,
          center[1] - length[1] / 2, 0.000000, 5.00e-7]

    right = ["Box_Ax", center[0] + length[0] / 2, center[1] - length[1] / 2 - width, center[0] + length[0] / 2 + width,
          center[1] + length[1] / 2 + width, -5.00e-7, 0.000000]
    left = ["Box_Ax", center[0] - length[0] / 2 - width, center[1] - length[1] / 2 - width, center[0] - length[0] / 2,
          center[1] + length[1] / 2 + width, 5.00e-7, 0.000000]
    ans = ""
    for i in [up,down,right,left]:
        line = ""
        for j in i:
            line = line + f'{j:.10}' + '\t'
        ans = ans + line + '\n'
    print(ans)

    with open('D:/Celegans-locomotion/software/input/coll_objs.tsv', 'w') as f:
        f.write(ans)
