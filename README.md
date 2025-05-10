## Complete app strructure is containerized 

```bash 

docker compose up

```




```bash 
python -m venv project_env

## for mac and linux 
source project_env/bin/activate

pip install -r requirements.txt


```

### Running a model using ollama
```bash 
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama      
docker exec -it ollama ollama run llama3.2:3b
```


### Enable Tracing (Optional)

```bash
docker pull arizephoenix/phoenix
docker run -p 6006:6006 -p 4317:4317 -i -t arizephoenix/phoenix:latest

# phoenix serve
```


### Run Streamlit 

```bash 

streamlit run app.py

```