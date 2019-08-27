import flask
from source.model.deep_tree_model import DeepTreeModel

# Initialize our Flask application and the Keras model
app = flask.Flask(__name__)


@app.route("/predict", methods=['POST'])
def predict():
    # Initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # Ensure an query_string was properly uploaded to our endpoint
    if flask.request.method == 'POST':
        query_string = flask.request.json

        preds = model.make_inference(query_string['input'])
        data['predictions'] = preds

        # Indicate that the request was a success
        data['success'] = True

    # Return the data dictionary as a JSON response
    return flask.jsonify(data)


if __name__ == '__main__':
    print("* Loading Keras model and Flask starting server..."
          "please wait until server has fully started")
    model = DeepTreeModel(hierarchy_level=4)
    app.run(debug=True, port=5005)
