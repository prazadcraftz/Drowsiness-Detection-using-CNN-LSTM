"""Reopen the candidate dashboard AVI and check SQLite against SessionMetrics."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys

import cv2

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.manifest import sha256,write_json


def validate(output):
    output=Path(output).resolve()
    summary=json.loads((output/'candidate_cli_summary.json').read_text(encoding='utf-8-sig'))
    path=output/'eye_close_60_lstm32.avi'
    source=cv2.VideoCapture(str(ROOT/'Eye Close/Eye Close_60.avi'))
    video=cv2.VideoCapture(str(path))
    def properties(c):
        return (int(c.get(cv2.CAP_PROP_FRAME_COUNT)),int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)),c.get(cv2.CAP_PROP_FPS))
    source_info,video_info=properties(source),properties(video)
    video.set(cv2.CAP_PROP_POS_FRAMES,55)
    ok,frame=video.read()
    if not ok:raise ValueError('Saved dashboard frame 55 unreadable')
    preview=output/'eye_close_60_preview.png'
    if not cv2.imwrite(str(preview),frame):raise ValueError('Preview write failed')
    video.set(cv2.CAP_PROP_POS_FRAMES,0)
    readable=all(video.read()[0] for _ in range(video_info[0]))
    source.release();video.release()
    if source_info!=video_info or not readable or video_info[0]!=summary['frames_decoded']:
        raise ValueError('Saved video properties/readability differ')
    write_json(output/'saved_video_validation.json',{'source':'Eye Close/Eye Close_60.avi',
        'output':'results/phase16/eye_close_60_lstm32.avi','source_properties':source_info,
        'saved_properties':video_info,'all_frames_readable':readable,
        'profile_and_readiness_display':'Visually inspect saved frame 55; Model: LSTM32 and Sequence: 32 / 32 are rendered',
        'preview':'results/phase16/eye_close_60_preview.png','no_render_exceptions':True,
        'output_sha256':sha256(path),'interpretation':'dashboard engineering validation only; folder name is not an accuracy annotation'})
    database=output/'candidate.db'
    with closing(sqlite3.connect(database.as_uri()+'?mode=ro',uri=True)) as connection:
        connection.row_factory=sqlite3.Row
        tables={row['name'] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        sessions=connection.execute('SELECT * FROM sessions').fetchall()
        if len(sessions)!=1:raise ValueError('Expected one candidate session')
        session=dict(sessions[0])
        metric=dict(connection.execute('SELECT * FROM session_metrics').fetchone())
        events=[dict(row) for row in connection.execute('SELECT * FROM events ORDER BY id')]
    expected=summary['session_metrics']
    if (json.loads(metric['aggregate_json'])!=expected or session['finalized']!=1 or not session['ended_at_utc']
        or session['model_name']!='LSTMOnly32' or session['parameter_count']!=209667 or session['sequence_length']!=32
        or session['checkpoint_sha256']!='1dcdcdbd17378190a6b7c429fa5ccf5b621dcf0adae6541d89d789a1d5b764ed'
        or session['normalization_sha256']!='9b397ae1327b515c5505610fffa37e198eb5ccd24ec7bf1d112eacec0f950edc'
        or tables.intersection({'frames','predictions'}) or len(events)!=sum(e['count'] for e in expected['events'].values())
        or any(e['ended_at_source_seconds'] is None for e in events)):
        raise ValueError('Candidate SQLite provenance/events/aggregate differ')
    write_json(output/'sqlite_validation.json',{'schema_version':1,'session_finalized':True,
        'model_name':session['model_name'],'parameter_count':session['parameter_count'],
        'sequence_length':session['sequence_length'],'checkpoint_sha256':session['checkpoint_sha256'],
        'normalization_sha256':session['normalization_sha256'],'aggregate_equals_session_metrics':True,
        'event_count':len(events),'events_equal_analytics_counts':True,
        'event_types':[e['event_type'] for e in events],'no_frame_or_prediction_tables':True,
        'tables':sorted(tables),'database_sha256':sha256(database)})
    print(video_info,'saved frames;',len(events),'SQLite events')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(ROOT/'results/phase16'))
    validate(parser.parse_args().output)
