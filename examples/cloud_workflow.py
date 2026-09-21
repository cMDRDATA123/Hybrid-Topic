"""Discover topics, export results, save, reload and assign new text.

Run from a checkout: python -m examples.cloud_workflow --output my_cloud_run
Uses OPENAI_API_KEY. Model weights must be cached unless --allow-download is set.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from hybrid_topic import HybridTopic, read_documents
from hybrid_topic.backends import OpenAITopicGenerator, SentenceTransformerEmbedder

DEMO_DOCUMENTS = [
    'The battery loses its charge before the end of the day.',
    'Battery life is excellent and one charge lasts two days.',
    'Charging the battery takes much longer than expected.',
    'The charging cable stopped working after a week.',
    'Delivery arrived three days late with no tracking update.',
    'Shipping was fast and the parcel arrived early.',
    'The parcel was delivered to the wrong address.',
    'The packaging protected the product during delivery.',
    'Customer support answered quickly and arranged a refund.',
    'Support has not replied to my refund request.',
    'The return process was simple and the refund arrived.',
    'The support agent helped me replace a faulty item.',
]


def run_workflow(documents, output, *, generator, embedder):
    output = Path(output)
    # Fail before paid generation if this destination already exists.
    output.mkdir(parents=True, exist_ok=False)
    model = HybridTopic(generator=generator, embedding_model=embedder)
    model.fit(documents)
    model.export(output / 'training_results')
    model.save(output / 'model')
    restored = HybridTopic.load(output / 'model', embedding_model=embedder)
    queries = ['The battery runs out quickly.', 'I am still waiting for my refund.']
    ids, scores = restored.transform(queries)
    for index, text in enumerate(queries):
        single = restored.transform([text])
        np.testing.assert_array_equal(single[0], ids[index:index+1])
        np.testing.assert_array_equal(single[1], scores[index:index+1])
    restored.transform_result(queries).export(output / 'new_document_results')
    history = model.metadata_['generation_history']
    summary = {'model': generator.generation_config['model'],
               'n_documents': len(documents), 'n_topics': len(model.get_topic_info()),
               'save_reload_batch_check': True,
               'generation_history': history}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--csv', type=Path, help='Omit to use 12 synthetic product comments')
    parser.add_argument('--text-column', default='text')
    parser.add_argument('--model', default='gpt-4.1-mini-2025-04-14')
    parser.add_argument('--context-window', type=int)
    parser.add_argument('--direction', help='Optional analysis direction')
    parser.add_argument('--embedding-model', default='Qwen/Qwen3-Embedding-0.6B')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--allow-download', action='store_true')
    args = parser.parse_args()
    if args.output.exists(): parser.error('--output must be a new directory')
    key = os.environ.get('OPENAI_API_KEY')
    if not key: parser.error('set OPENAI_API_KEY in the environment')
    texts = read_documents(args.csv, text_column=args.text_column) if args.csv else DEMO_DOCUMENTS
    generator = OpenAITopicGenerator(model=args.model, api_key=key,
        context_window=args.context_window, topic_instruction=args.direction)
    embedder = SentenceTransformerEmbedder(args.embedding_model, device=args.device,
        local_files_only=not args.allow_download)
    model = run_workflow(texts, args.output, generator=generator, embedder=embedder)
    print(model.get_topic_info()[['topic_id','name','count']].to_string(index=False))
    print('Saved model and query batch checks passed. Outputs:', args.output.resolve())

if __name__ == '__main__': main()
