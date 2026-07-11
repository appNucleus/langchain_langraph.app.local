def after_verification(state: dict) -> str:
    verdict = (state.get('verification') or {}).get('verdict', 'revise')
    if verdict == 'pass':
        return 'advance'
    if verdict == 'research':
        return 'research'
    if verdict == 'replan':
        return 'replan'
    return 'revise'

def after_advance(state: dict) -> str:
    tasks = (state.get('plan') or {}).get('tasks', [])
    return 'finalize' if state.get('task_index', 0) >= len(tasks) else 'worker'
