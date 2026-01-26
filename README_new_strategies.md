Esta rama contiene las nuevas estrategias de ensamblaje propuestas. Para usar esta rama:

```bash
git clone https://github.com/sofiaaduarte/ET-Pfam-sandbox.git
cd ET-Pfam-sandbox
git checkout ensemble_strategy
```

Este branch incluye nuevas estrategias de ensamblaje utilizando capas `Linear` de PyTorch:
- `family_linear`: *Learned weights by family (LWF) perceptron voting*. Esta es la misma que la LWF original pero implementada con una capa `Linear` de PyTorch.
- `family_mlp_linear`: *Learned weights by family MLP voting*
- `flatten_linear`: *Learned stacking perceptron voting*
- `flatten_mlp`: *Learned stacking MLP voting*

Las estrategias pueden ser entrenadas y probadas individualmente usando el script `train_test_ensemble.py`.

Para comenzar a testear las nuevas estrategias, se requieren todos los archivos de datos utilizados originalmente en el proyecto, los embeddings, como así también los modelos pre-entrenados. Por lo tanto, es necesario seguir los pasos planteados en el repositorio original para obtener los archivos necesarios.

# Configuración de los modelos
Dado que esta rama introduce, además de las estrategias de ensamblaje, la posibilidad de ensamblar modelos base entrenados a partir de diferentes pLMs, es necesario modificar los archivos de configuración, para indicar el path de los embeddings correspondientes. Esto se hace con el script `change_embeddings_path.py`, que recibe los siguientes argumentos:
- `-m` o `--models_path`: path donde se encuentran los modelos base entrenados (podría ser `models/mini/` o `models/full/`)
- `-e` o `--embeddings_path`: path donde se encuentran los embeddings correspondientes al pLM utilizado para entrenar los modelos base.
- `-p` o `--plm`: nombre del pLM utilizado para entrenar los modelos base. Actualmente, las opciones disponibles son `esm2` y `ptt5`.
- `-f` o `--filter`: filtro para seleccionar qué modelos modificar. Los nombres de las carpetas de los modelos base deberían contener el nombre del pLM utilizado para entrenar los modelos. Si los archivos se descargaron del repositorio original, este argumento no es necesario, ya que el script aplica el cambio automáticamente a todos los modelos disponibles.

Entonces, si desea ensamblar los modelos base para el mini-dataset, entrenados con ESM2 y descargados de la manera en la cual se indica en el repositorio original, y se conoce que los embeddings de ESM2 están en la carpeta `/home/data/pfam32/embeddings/esm2/`, es necesario ejecutar el siguiente comando:

```bash
python3 change_embeddings_path.py -m models/mini/ -e /home/data/pfam32/embeddings/esm2/ -p esm2
```

# Ejecución de experimentos con nuevas estrategias de ensamblaje
Para entrenar y probar las nuevas estrategias de ensamblaje, es necesario modificar el archivo `config/ensemble.json` para indicar la estrategia deseada y los hiperparámetros correspondientes. Por ejemplo, para entrenar y probar la estrategia `flatten_mlp` con bias, tamaño oculto de 1024 y 500 épocas, el archivo `config/ensemble.json` debería quedar así:

```json
{
    "voting_strategy": "flatten_mlp",
    "use_bias": true,
    "hidden_size": 1024,
    "learning_rate": 0.01,
    "n_epochs": 500
}
```
Después, se puede ejecutar el script `train_test_ensemble.py` para entrenar y probar la estrategia seleccionada. Por ejemplo, para el mini-dataset:

```bash
python3 train_test_ensemble.py -m models/mini/
``` 

Para el full, se puede ejecutar:

```bash
python3 train_test_ensemble.py -m models/full/
```

También es posible realizar el ensemble de modelos base entrenados con diferentes pLMs. En ese caso, colocar en una carpeta nueva los modelos a ensamblar (por ejemplo, `models/full_mixed/`), modificar (si se requiere) los archivos de configuración de cada modelo base con el script `change_embeddings_path.py`, y luego ejecutar el script `train_test_ensemble.py` indicando la nueva carpeta:

```bash
python3 train_test_ensemble.py -m models/full_mixed/
``` 