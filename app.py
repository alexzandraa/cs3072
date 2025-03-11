from flask import Flask, render_template, session, request, redirect, url_for
import pandas as pd
import joblib

app = Flask(__name__)


@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/stage1')
def stage1():
    stage_title = "Stage 1: Accurate, Fair, and Transparent"
    objective = ("Establish a baseline for trust by presenting an AI system that is accurate, unbiased, "
                 "and fully transparent in its decision-making process. Participants are provided with detailed explanations for every decision, "
                 "allowing them to assess both the outcomes and the reasoning.")
    research_question = "Does a fair and transparent AI system enhance user trust and understanding of its decisions?"
    purpose = ("This stage serves as a control condition, representing the ideal performance of an AI system in terms of accuracy, fairness, and transparency.")
    return render_template('stage.html', stage_title=stage_title, objective=objective,
                           research_question=research_question, purpose=purpose)

@app.route('/stage2')
def stage2():
    stage_title = "Stage 2: Accurate but Opaque"
    objective = ("Test the impact of removing transparency on trust while retaining the accuracy and fairness of the AI system. "
                 "Participants see only the outcomes without explanations.")
    research_question = "Does the lack of transparency affect participants’ perception of trust and fairness, even when the system remains accurate and unbiased?"
    purpose = ("This stage reflects real-world challenges where AI systems may lack interpretability, highlighting the importance of transparency.")
    return render_template('stage.html', stage_title=stage_title, objective=objective,
                           research_question=research_question, purpose=purpose)

# -------------------------------
# Stage 3: Biased and Opaque (Interactive Candidate Review)
# -------------------------------
@app.route('/stage3', methods=['GET', 'POST'])
def stage3():
    stage_title = "Stage 3: Biased but Opaque"
    objective = ("Observe how participants react when the AI system is biased and provides no explanations")
    research_question = "Does the lack of transparency affect participants’ perception of trust and fairness, even when the system remains accurate and unbiased?"
    purpose = ("This stage reflects real-world challenges where AI systems may lack interpretability, highlighting the importance of transparency.")
    return render_template('stage.html', stage_title=stage_title, objective=objective,
                           research_question=research_question, purpose=purpose)

    

@app.route('/stage4')
def stage4():
    stage_title = "Stage 4: Biased but Transparent"
    objective = ("Observe participants’ reactions when the AI system openly displays its biased decision-making. "
                 "The system remains biased but provides full transparency, including explanations that reveal the influence of biased factors.")
    research_question = ("Does transparency mitigate the negative impact of bias on user trust? "
                         "Does revealing the bias change participants’ perception of fairness?")
    purpose = ("This stage examines whether transparency alone is sufficient to rebuild trust in a biased system, or if inherent bias continues to erode confidence.")
    return render_template('stage.html', stage_title=stage_title, objective=objective,
                           research_question=research_question, purpose=purpose)

if __name__ == '__main__':
    app.run(debug=True)
