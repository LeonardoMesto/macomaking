import gradio as gr
import numpy as np
from functions import prediction, printPlayers
from loadData import getData, getModel

solo, duo, trio, squad, users = getData()

def greet(type, score, top1, kd, winRatio, matches, kills, minutes):
    model = getModel(type)

    p1 = np.array(model.transform([[score, top1, kd, winRatio, matches, kills, minutes]])[0])

    if type == 'Solo': userList, fig = prediction(p1, solo, 1, users)
    if type == 'Duo': userList, fig = prediction(p1, duo, 3, users)
    if type == 'Trio': userList, fig =  prediction(p1, trio, 5, users)
    if type == 'Squad': userList, fig =  prediction(p1, squad, 7, users)

    userList = printPlayers(userList)

    return userList, fig

demo = gr.Interface(
    fn=greet,
    examples = [
        ["Solo",1136282,1400,6.32,30,4429,19591,36245],
        ["Duo",1010165,900,5.23,24.3,3497,13846,35857],
        ["Trio",612525.0,209.0,2,8.6,2421.0,2526.0,27403.0],
        ["Squad",599549.0,321.0,2.86,8,4117.0,10842.0,25374.0]
    ],
    inputs=[
        gr.Dropdown(["Solo", "Duo", "Trio", "Squad"]),
        "number",
        "number",
        "number",
        "number",
        "number",
        "number",
        "number"
    ],
    outputs=[gr.Text(label= "Similar Players"), gr.Plot()]
)

demo.launch()