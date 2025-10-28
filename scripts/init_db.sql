-- Schema para TradeSigns PoC Live

-- Extensão TimescaleDB para séries temporais
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Tabela de preços em tempo real
CREATE TABLE IF NOT EXISTS live_prices (
    id BIGSERIAL,
    pair VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    ticks INTEGER DEFAULT 0,
    PRIMARY KEY (pair, timestamp)
);

-- Converter para hypertable (TimescaleDB)
SELECT create_hypertable('live_prices', 'timestamp', if_not_exists => TRUE);

-- Índices para performance
CREATE INDEX IF NOT EXISTS idx_live_prices_pair_time ON live_prices (pair, timestamp DESC);

-- Tabela de sinais
CREATE TABLE IF NOT EXISTS signals (
    id BIGSERIAL PRIMARY KEY,
    pair VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    signal_type VARCHAR(10) NOT NULL, -- BUY, SELL
    strategy VARCHAR(50) NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    score DECIMAL(5, 4), -- 0.0 a 1.0
    indicators JSONB, -- Snapshot dos indicadores
    status VARCHAR(20) DEFAULT 'OPEN', -- OPEN, CLOSED
    exit_price DECIMAL(20, 8),
    exit_timestamp TIMESTAMPTZ,
    pnl_pct DECIMAL(10, 6),
    outcome VARCHAR(20), -- WIN, LOSS, TIMEOUT
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signals_pair_time ON signals (pair, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals (status);
CREATE INDEX IF NOT EXISTS idx_signals_created ON signals (created_at DESC);

-- Tabela de estatísticas agregadas (cache)
CREATE TABLE IF NOT EXISTS performance_stats (
    id SERIAL PRIMARY KEY,
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    total_signals INTEGER,
    wins INTEGER,
    losses INTEGER,
    timeouts INTEGER,
    win_rate DECIMAL(5, 4),
    total_pnl DECIMAL(10, 6),
    avg_pnl DECIMAL(10, 6),
    max_drawdown DECIMAL(10, 6),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(period_start, period_end)
);

-- View para estatísticas das últimas 24h
CREATE OR REPLACE VIEW stats_24h AS
SELECT 
    COUNT(*) as total_signals,
    COUNT(*) FILTER (WHERE outcome = 'WIN') as wins,
    COUNT(*) FILTER (WHERE outcome = 'LOSS') as losses,
    COUNT(*) FILTER (WHERE outcome = 'TIMEOUT') as timeouts,
    ROUND(
        COUNT(*) FILTER (WHERE outcome = 'WIN')::DECIMAL / 
        NULLIF(COUNT(*) FILTER (WHERE status = 'CLOSED'), 0), 
        4
    ) as win_rate,
    ROUND(SUM(pnl_pct), 6) as total_pnl,
    ROUND(AVG(pnl_pct), 6) as avg_pnl
FROM signals
WHERE created_at >= NOW() - INTERVAL '24 hours';

-- Inserir dados iniciais (exemplo)
INSERT INTO performance_stats (period_start, period_end, total_signals, wins, losses, timeouts, win_rate, total_pnl, avg_pnl, max_drawdown)
VALUES (NOW() - INTERVAL '24 hours', NOW(), 0, 0, 0, 0, 0, 0, 0, 0)
ON CONFLICT (period_start, period_end) DO NOTHING;

-- Log de inicialização
DO $$
BEGIN
    RAISE NOTICE 'TradeSigns database schema initialized successfully!';
END $$;
