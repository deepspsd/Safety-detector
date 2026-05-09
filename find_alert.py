content = open(r'backend\routers\cctv.py', 'r', encoding='utf-8').read()
lf = content.replace('\r\n', '\n')
idx = lf.find('save_alert(')
if idx >= 0:
    print('save_alert found at char', idx)
    print(repr(lf[max(0,idx-200):idx+600]))
else:
    print('save_alert call not found in cctv.py — checking confidence references:')
    for i, line in enumerate(lf.split('\n'), 1):
        if 'confidence' in line.lower() or 'alert' in line.lower():
            print(f'  L{i}: {line[:120]}')
