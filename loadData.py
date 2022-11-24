from joblib import load

solo_model, duo_model, trio_model, squad_model, ltm_model = load('minMaxModels.joblib')

def getData():
    solo = load('solo.joblib')
    duo = load('duo.joblib')
    trio = load('trio.joblib')
    squad = load('squad.joblib')
    users = load('users.joblib')
    return solo, duo, trio, squad, users

def getModel(type):
    if type == 'Solo': return solo_model
    if type == 'Duo': return duo_model
    if type == 'Trio': return trio_model
    if type == 'Squad': return squad_model