CREATE TABLE public.wrist_samples (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  device_id text NOT NULL,
  session_id text NOT NULL,
  timestamp_ms bigint NOT NULL,
  theta_fe numeric,
  theta_rud numeric,
  pressure_kpa numeric,
  temperature_c numeric,
  battery_pct smallint,
  quality numeric,
  raw jsonb,
  received_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX wrist_samples_session_ts_idx ON public.wrist_samples (session_id, timestamp_ms);
CREATE INDEX wrist_samples_device_ts_idx ON public.wrist_samples (device_id, timestamp_ms);

GRANT SELECT ON public.wrist_samples TO authenticated;
GRANT ALL ON public.wrist_samples TO service_role;

ALTER TABLE public.wrist_samples ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Authenticated users can read samples" ON public.wrist_samples
  FOR SELECT TO authenticated USING (true);

ALTER PUBLICATION supabase_realtime ADD TABLE public.wrist_samples;