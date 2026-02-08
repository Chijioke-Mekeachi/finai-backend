from datetime import date, datetime
from pydantic import BaseModel, EmailStr, Field


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: str
    email: EmailStr
    name: str
    subscription_plan_id: str


class AuthSignup(BaseModel):
    email: EmailStr
    # bcrypt has a 72-byte limit; backend uses bcrypt_sha256 so long passwords are OK,
    # but we still cap size to prevent abuse.
    password: str = Field(min_length=8, max_length=1024)
    name: str
    currency: str | None = None


class AuthLogin(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=1024)


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str = Field(min_length=8)


class PasswordUpdate(BaseModel):
    new_password: str = Field(min_length=8)


class TransactionCreate(BaseModel):
    date: date
    type: str
    category: str
    amount: float
    description: str | None = None
    entity: str


class TransactionOut(TransactionCreate):
    id: str


class CompanyGoalBase(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    description: str | None = None
    unit: str | None = Field(default=None, max_length=20)
    target_value: float
    current_value: float = 0
    start_date: date | None = None
    due_date: date | None = None
    status: str = Field(default="active", max_length=20)  # active|completed|archived


class CompanyGoalCreate(CompanyGoalBase):
    pass


class CompanyGoalUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
    unit: str | None = Field(default=None, max_length=20)
    target_value: float | None = None
    current_value: float | None = None
    start_date: date | None = None
    due_date: date | None = None
    status: str | None = Field(default=None, max_length=20)


class CompanyGoalOut(CompanyGoalBase):
    id: str
    created_at: datetime
    updated_at: datetime


class BusinessSettingsOut(BaseModel):
    company_name: str
    currency: str
    fiscal_year_start: str
    tax_rate: float


class BusinessSettingsUpsert(BusinessSettingsOut):
    pass


class SubscriptionOut(BaseModel):
    plan_id: str


class SubscriptionUpdate(BaseModel):
    plan_id: str


class AiAnalyzeRequest(BaseModel):
    prompt: str
    # Model is optional; backend will fall back to GEMINI_MODEL env var.
    model: str | None = None
    purpose: str | None = None
    # Optional: the raw user message (without system instructions). If provided,
    # this is what will be persisted as the "user" chat message.
    user_message: str | None = None


class AiAnalyzeResponse(BaseModel):
    text: str


class AiVisionRequest(BaseModel):
    prompt: str
    image_base64: str
    mime_type: str = "image/jpeg"
    # Model is optional; backend will fall back to GEMINI_MODEL env var.
    model: str | None = None
    purpose: str | None = None
    user_message: str | None = None


class AiTtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    # Model is optional; backend will fall back to GEMINI_TTS_MODEL env var.
    model: str | None = None
    # Optional: prebuilt voice name, e.g. "Kore".
    voice_name: str | None = None
    # Optional BCP-47 language tag, e.g. "en-US".
    language_code: str | None = None
    # Optional: label for analytics/logging.
    purpose: str | None = None
    # Optional: style prefix like "Say calmly and professionally:".
    style: str | None = None


class AiTtsResponse(BaseModel):
    audio_base64: str
    mime_type: str = "audio/wav"
    sample_rate_hz: int = 24000


class AdminUserSummary(BaseModel):
    id: str
    email: EmailStr
    name: str
    subscription_plan_id: str
    created_at: datetime
    dummy_password: str = "********"
    transactions_count: int
    settings: BusinessSettingsOut | None = None


class AdminUserDetail(AdminUserSummary):
    transactions: list[TransactionOut] = []


class AdminTransactionRow(BaseModel):
    id: str
    user_id: str
    user_email: EmailStr
    user_name: str
    user_plan_id: str
    date: date
    type: str
    category: str
    amount: float
    entity: str
    description: str | None = None
    created_at: datetime


class AiMessageOut(BaseModel):
    id: str
    role: str
    purpose: str
    content: str
    created_at: datetime


class AiChatSummaryOut(BaseModel):
    id: str
    purpose: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages_count: int


class PaystackInitializeRequest(BaseModel):
    plan_id: str = Field(min_length=1, max_length=50)
    callback_url: str | None = None


class PaystackInitializeResponse(BaseModel):
    authorization_url: str
    access_code: str
    reference: str


class PaystackVerifyResponse(BaseModel):
    status: str  # succeeded|failed
    plan_id: str | None = None
    reference: str
