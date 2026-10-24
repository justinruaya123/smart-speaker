"""Current model cards link moved application source to its actual owner."""
import pytest
from scripts.iris_publish import published_links


@pytest.mark.parametrize('href',[
    'SpeakerUI/src/App.svelte',
    '../SpeakerUI/src/App.svelte',
    'https://huggingface.co/justinruaya123/ai231-smart-speaker/blob/main/reproduction/source/SpeakerUI/src/App.svelte',
])
def test_model_publication_links_application_to_consumer(href):
    assert published_links('[application]('+href+')') == (
        '[application](https://github.com/justinruaya123/SpeakerUI/blob/main/src/App.svelte)')


def test_model_publication_links_training_to_producer():
    assert published_links('[training](scripts/train/iris_owner.py)') == (
        '[training](https://github.com/justinruaya123/smart-speaker/blob/main/scripts/train/iris_owner.py)')
