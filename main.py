from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector_store import retriever
from flask import Flask, render_template, request

model = OllamaLLM(model="llama3.2")

template = """
You are an exeprt in answering questions about a Control Plan and Inspection Plan Module which is in Teamcenter.
Here are information about Control Plan and Inspection Plan Module: {data}
Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Start flask app and set to ngrok
app = Flask(__name__)

@app.route('/')
def initial():
  return render_template('index.html', input="", response="")

@app.route('/submit-prompt', methods=['POST'])
def generate_response():
  prompt = request.form['prompt-input']
  result = ""
  if prompt.strip(): 
    data = retriever.invoke(prompt)
    result = chain.invoke({"data": data, "question": prompt})
    return render_template('index.html', input=prompt, response=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

    