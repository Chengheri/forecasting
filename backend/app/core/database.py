from sqlalchemy import create_engine, Column, Integer, Float, DateTime, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from ..utils.logger import Logger

load_dotenv()

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./electricity_forecasting.db")

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

logger = Logger()

class ConsumptionData(Base):
    __tablename__ = "consumption_data"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False)
    consumption = Column(Float, nullable=False)
    temperature = Column(Float)
    humidity = Column(Float)
    is_holiday = Column(Integer)
    day_type = Column(String)  # workday, weekend, holiday

    def __repr__(self):
        return f"<ConsumptionData(date={self.timestamp}, value={self.consumption}, source={self.day_type})>"

class ForecastResult(Base):
    __tablename__ = "forecast_results"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, nullable=False)
    forecast_timestamp = Column(DateTime, nullable=False)
    forecast_value = Column(Float, nullable=False)
    confidence_lower = Column(Float)
    confidence_upper = Column(Float)
    metrics = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ForecastResult(model={self.model_name}, date={self.forecast_timestamp}, value={self.forecast_value})>"

class Database:
    def __init__(self, connection_string: str):
        logger.info(f"Initializing database connection with: {connection_string}")
        self.engine = create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine)
        
    def create_tables(self):
        """Create all database tables if they don't exist."""
        logger.info("Creating database tables")
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created successfully")
        
    def add_consumption_data(self, data: list):
        """Add consumption data to the database."""
        logger.info(f"Adding {len(data)} consumption data records")
        session = self.Session()
        try:
            for record in data:
                consumption = ConsumptionData(
                    timestamp=record['date'],
                    consumption=record['value'],
                    temperature=record.get('temperature'),
                    humidity=record.get('humidity'),
                    is_holiday=record.get('is_holiday'),
                    day_type=record.get('day_type')
                )
                session.add(consumption)
            session.commit()
            logger.info("Consumption data added successfully")
        except Exception as e:
            logger.error(f"Error adding consumption data: {str(e)}")
            session.rollback()
            raise
        finally:
            session.close()
            
    def get_consumption_data(self, start_date: datetime, end_date: datetime):
        """Retrieve consumption data for a date range."""
        logger.info(f"Retrieving consumption data from {start_date} to {end_date}")
        session = self.Session()
        try:
            data = session.query(ConsumptionData).filter(
                ConsumptionData.timestamp.between(start_date, end_date)
            ).all()
            logger.info(f"Retrieved {len(data)} consumption data records")
            return data
        except Exception as e:
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create all tables
Base.metadata.create_all(bind=engine) 